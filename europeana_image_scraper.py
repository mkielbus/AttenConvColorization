"""Europeana scraper for the Phase 2 dataset expansion.

Reworked from the original grayscale collector: the v1/v2 dataset was built by KEEPING grayscale
images, but the expansion needs COLOUR images, because they are the colorization targets.

The hard part is not grayscale (trivially detectable) but sepia and duotone scans: they are not
grayscale, and they carry HIGHER mean saturation than real photographs (measured: 0.46 vs a real
median of 0.34), so any saturation threshold admits them preferentially. The discriminator is the
circular spread of hue, measured after a blur -- see analysis/phase2_calibrate_color_filter.py for
the calibration against the existing dataset and against synthetic sepia at several noise levels.

Both thresholds are set at roughly the p1-p5 of `dataset_flat_v2/train`, because that dataset IS
the target distribution: admitting images the existing val/test sets do not contain would shift
the very balance Phase 2 exists to preserve. Measured on 400 train images + synthetic sepia:

    rule                        real kept   sepia   sepia+2% noise   sepia+5% noise
    spread>=5,  chrom>=0.05       98.3%     24.7%       50.0%            84.7%
    spread>=12, chrom>=0.25       89.5%      1.3%        6.7%            17.3%

The 9% of real images the strict rule loses are mostly single-hue paintings and objects on white
backgrounds -- an acceptable price, since Europeana holds millions of candidates but a sepia scan
admitted as "colour" teaches the model to predict sepia.

Downloaded images are resized to a 1200 px long side (matching the existing data) and recorded in
a manifest CSV with their Europeana record IDs, so provenance survives into split v3.
"""

import argparse
import csv
import faulthandler
import os
import signal
import socket
import sys
import threading
import time
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from tqdm import tqdm

SEARCH_URL: str = "https://api.europeana.eu/record/v2/search.json"

# --- Colour admission thresholds (calibrated, see module docstring) -------------------------
SATURATION_THRESHOLD: float = 0.20     # extract_colors.py's chromatic/achromatic boundary
BLUR_KERNEL: int = 9                   # noise in a degraded scan fakes hue variety; blur it out
MIN_HUE_SPREAD_DEG: float = 12.0       # dataset_flat_v2 train: p1=4.9, p5=10.6, median 45.6
MIN_CHROMATIC_FRACTION: float = 0.25   # dataset_flat_v2 train: p1=0.267, median 0.709
GRAYSCALE_TOLERANCE: float = 2.0       # mean |R-G|,|G-B| below this = grayscale saved as RGB
LONG_SIDE: int = 1200

# --- Network limits (see downloadBytes) -----------------------------------------------------
REQUEST_TIMEOUT: Tuple[float, float] = (10.0, 20.0)   # (connect, read) -- per socket operation
MAX_SECONDS_PER_IMAGE: float = 45.0                   # hard wall-clock cap per download
MAX_IMAGE_BYTES: int = 64 * 1024 * 1024
CHUNK_BYTES: int = 64 * 1024
SEARCH_TIMEOUT: Tuple[float, float] = (10.0, 30.0)
SEARCH_RETRIES: int = 5
SOCKET_TIMEOUT: float = 30.0                          # secondary guard (see main)
ABANDON_SECONDS: float = 90.0                         # wall clock per record before giving up
POLL_SECONDS: float = 5.0
HEARTBEAT_SECONDS: float = 30.0
HOST_STRIKES: int = 5                                 # stalls/timeouts before skipping a host


def isGrayscale(bgr: np.ndarray, tolerance: float = GRAYSCALE_TOLERANCE) -> bool:
    """True when the channels are (near) identical -- a grayscale image stored as RGB.

    A tolerance is used rather than exact equality because JPEG compression perturbs channels
    slightly, so an exact test misses most real grayscale scans.
    """
    blue, green, red = cv2.split(bgr.astype(np.float32))
    return (float(np.abs(red - green).mean()) < tolerance
            and float(np.abs(green - blue).mean()) < tolerance)


def hueSpreadDegrees(bgr: np.ndarray, saturation_threshold: float = SATURATION_THRESHOLD,
                     blur_kernel: int = BLUR_KERNEL) -> Tuple[float, float]:
    """Circular standard deviation of hue over chromatic pixels, plus the chromatic fraction.

    Returns (hue_spread_deg, chromatic_fraction). A single-hue image (sepia, duotone, a colour
    cast over a scan) has a spread near zero; a real colour photograph's median is ~43 degrees.
    """
    if blur_kernel > 1:
        bgr = cv2.GaussianBlur(bgr, (blur_kernel, blur_kernel), 0)
    hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue: np.ndarray = hsv[..., 0].astype(np.float64) * 2.0
    saturation: np.ndarray = hsv[..., 1].astype(np.float64) / 255.0

    chromatic: np.ndarray = saturation >= saturation_threshold
    fraction: float = float(chromatic.mean())
    if chromatic.sum() < 16:
        return 0.0, fraction
    radians: np.ndarray = np.deg2rad(hue[chromatic])
    resultant: float = float(np.hypot(np.cos(radians).mean(), np.sin(radians).mean()))
    resultant = min(max(resultant, 1e-9), 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))), fraction


def isColorImage(bgr: np.ndarray) -> Tuple[bool, str]:
    """Admission test. Returns (accepted, reason) -- the reason is logged for auditability."""
    if isGrayscale(bgr):
        return False, "grayscale"
    spread, fraction = hueSpreadDegrees(bgr)
    if fraction < MIN_CHROMATIC_FRACTION:
        return False, f"neutral(chrom={fraction:.3f})"
    if spread < MIN_HUE_SPREAD_DEG:
        return False, f"monochrome(hue_spread={spread:.1f})"
    return True, f"ok(hue_spread={spread:.1f},chrom={fraction:.2f})"


def resizeLongSide(bgr: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    """Downscale so the longer side is `long_side`; never upscales (that invents detail)."""
    height, width = bgr.shape[:2]
    longest: int = max(height, width)
    if longest <= long_side:
        return bgr
    scale: float = long_side / longest
    return cv2.resize(bgr, (int(round(width * scale)), int(round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def loadManifest(path: Path) -> Tuple[Set[str], int, int]:
    """Record IDs seen, next free index, and how many images were ALREADY accepted.

    The accepted count is what makes `--number_of_images` mean the same thing on a resumed run as
    on a fresh one. Without it a resume starts counting from zero and fetches a second full target
    on top of the images already on disk.
    """
    if not path.exists():
        return set(), 0, 0
    seen: Set[str] = set()
    next_index: int = 0
    accepted: int = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            seen.add(row["record_id"])
            if row["status"] == "accepted":
                accepted += 1
            if row.get("index"):                 # rejected rows carry no index
                next_index = max(next_index, int(row["index"]) + 1)
    return seen, next_index, accepted


def searchPage(params: Dict[str, object]) -> Optional[dict]:
    """One search page, with a bounded timeout and retries. None means 'stop paging'.

    This runs on the MAIN thread, so a hang here freezes the whole scrape -- no downloads are
    submitted and no results are collected. It happened twice: `api.europeana.eu` sits behind
    Cloudflare and normally answers in 0.16 s, but a single connection that stalls mid-response
    blocks forever under a per-socket timeout, because that clock restarts on every byte.
    """
    for attempt in range(SEARCH_RETRIES):
        try:
            response = requests.get(SEARCH_URL, params=params, timeout=SEARCH_TIMEOUT)
        except Exception as error:
            tqdm.write(f"search retry {attempt + 1}/{SEARCH_RETRIES}: {type(error).__name__}")
            time.sleep(2.0 ** attempt)
            continue
        if response.status_code == 200:
            return response.json()
        tqdm.write(f"search HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code < 500:               # a bad query will not fix itself
            return None
        time.sleep(2.0 ** attempt)
    tqdm.write(f"search: {SEARCH_RETRIES} nieudanych prob -- przerywam strumien rekordow")
    return None


def streamRecords(query: str, qf: List[str], theme: str, wskey: str,
                  limit: int) -> Iterator[Dict[str, str]]:
    """Page through the search API with a cursor, yielding (record_id, image_url) pairs.

    A generator rather than a list because the caller stops once enough images have been
    *accepted*: how many records that takes is only known while downloading, and pre-fetching a
    fixed multiple of the target would either overshoot (wasted downloads) or fall short.
    """
    yielded: int = 0
    params: Dict[str, object] = {"query": query, "qf": qf, "theme": theme, "media": "true",
                                 "wskey": wskey, "cursor": "*", "rows": 100}
    while yielded < limit and params["cursor"]:
        data: Optional[dict] = searchPage(params)
        if data is None:
            return
        items: List[dict] = data.get("items", [])
        if not items:
            return
        for item in items:
            shown_by: List[str] = item.get("edmIsShownBy") or []
            if not shown_by:
                continue
            yield {"record_id": item.get("id", ""), "image_url": shown_by[0]}
            yielded += 1
            if yielded >= limit:
                return
        params["cursor"] = data.get("nextCursor", "")


_thread_local: threading.local = threading.local()


def installConnectionRecorder() -> bool:
    """Make each worker publish the urllib3 connection it is currently using.

    Abandoning a future leaves its thread blocked in a socket read forever, holding a thread and a
    file descriptor for the life of the process. The only way to free them is to close the socket
    from the outside, and to do that the main thread needs a handle on it.

    `_make_request` is the hook because it runs for EVERY request and receives the live connection
    -- wrapping `connect()` would miss connections reused from the pool, which is most of them.
    It is urllib3-internal, so a version that renames it degrades to the previous behaviour
    (abandon without closing) instead of breaking the scrape.
    """
    try:
        from urllib3.connectionpool import HTTPConnectionPool
        original = HTTPConnectionPool._make_request
    except (ImportError, AttributeError):
        return False

    def recordingMakeRequest(self, conn, *args, **kwargs):        # type: ignore[no-untyped-def]
        handle: Optional[Dict[str, object]] = getattr(_thread_local, "handle", None)
        if handle is not None:
            handle["conn"] = conn
        return original(self, conn, *args, **kwargs)

    HTTPConnectionPool._make_request = recordingMakeRequest
    return True


def closeHandle(handle: Dict[str, object]) -> bool:
    """Break the connection a stuck worker is blocked on, so its thread and fd come back.

    `close()` alone does NOT do this: it drops a reference, while the other thread stays parked in
    `recv` on the same descriptor. `shutdown()` is what tears the connection down underneath it and
    makes that read raise. Measured on a dribbling TLS server: with `close()` only, 24 abandoned
    records leaked 25 threads and 25 descriptors; with `shutdown()` first, nothing leaks.
    """
    conn = handle.get("conn")
    if conn is None:
        return False                       # never got as far as a connection
    released: bool = False
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
            released = True
        except OSError:
            pass                           # already torn down by the peer
    try:
        conn.close()
        released = True
    except Exception:
        pass
    return released


def workerSession() -> requests.Session:
    """One Session per worker thread.

    `requests.Session` is not documented as thread-safe, and sharing one across 16 threads that
    all hold streamed responses puts its connection pool in a state that is hard to reason about.
    A session per thread costs a few sockets and removes the question entirely.
    """
    session: Optional[requests.Session] = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def downloadBytes(url: str, session: requests.Session) -> bytes:
    """Fetch a URL under a HARD wall-clock deadline and a size cap.

    `requests`' timeout is per socket operation, not per request: a server that trickles one byte
    before every timeout window holds the connection open forever. That is not hypothetical -- it
    stalled a 40k-image scrape here, and because the consumer collects results in FIFO order, one
    stuck worker froze all of them. Reading in chunks against a monotonic deadline is what bounds
    the wait; the size cap stops a mislabelled multi-GB TIFF from doing the same thing slowly.
    """
    chunks: List[bytes] = []
    total: int = 0
    deadline: float = time.monotonic() + MAX_SECONDS_PER_IMAGE
    with session.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(f"http{response.status_code}")
        for chunk in response.iter_content(CHUNK_BYTES):
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise RuntimeError("too_big")
            if time.monotonic() > deadline:
                raise RuntimeError("deadline")
    return b"".join(chunks)


def fetchAndFilter(record: Dict[str, str],
                   handle: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Download one record, decode it and run the admission test. Runs in a worker thread.

    Everything that can fail (network, decode, filter) happens here; the caller only writes.
    The image is already resized, so worker threads carry the resize cost too.
    """
    result: Dict[str, object] = {"record": record, "status": "failed", "reason": "",
                                 "image": None}
    _thread_local.handle = handle        # lets installConnectionRecorder publish the connection
    try:
        content: bytes = downloadBytes(record["image_url"], workerSession())
        buffer: np.ndarray = np.frombuffer(content, np.uint8)
        bgr: Optional[np.ndarray] = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as error:
        result["reason"] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        return result
    if bgr is None or bgr.size == 0:
        result["reason"] = "decode"
        return result

    accepted, reason = isColorImage(bgr)
    result["reason"] = reason
    result["status"] = "accepted" if accepted else "rejected"
    result["image"] = resizeLongSide(bgr)
    return result


def strikeHost(host: str, strikes: Dict[str, int], blocked_hosts: Set[str]) -> None:
    """Count one stall/failure against a host and block it once it has used up its strikes."""
    if not host or host in blocked_hosts:
        return
    strikes[host] = strikes.get(host, 0) + 1
    if strikes[host] >= HOST_STRIKES:
        blocked_hosts.add(host)
        tqdm.write(f"host {host} nie odpowiada ({strikes[host]}x) -- pomijam jego rekordy")


def downloadRecords(records: Iterator[Dict[str, str]], out_dir: Path, manifest_path: Path,
                    category: str, query: str, target_accepted: int, workers: int = 8,
                    max_attempts: int = 0,
                    rejected_dir: Optional[Path] = None) -> Dict[str, int]:
    """Fetch, filter and save until `target_accepted` images are on disk (or records run out).

    Files are named {category}_n{idx}.jpg so they cannot collide with the existing
    {category}_{idx}.jpg dataset.

    Downloads run in a thread pool -- they are almost entirely waiting on provider hosts, and a
    40k-image scrape is ~10 h sequentially. Only saving and manifest writing happen on the main
    thread, which keeps index assignment and the CSV serialized without a lock.

    A worker can block FOREVER and there is no way to interrupt it from Python: a server that
    dribbles one byte at a time satisfies every socket read, so neither the requests timeout nor
    `socket.setdefaulttimeout` ever fires, and the thread sits in `getresponse()` indefinitely
    (verified against a deliberately dribbling TLS server). This is not theoretical -- it stalled
    this scrape twice. So the wall clock is enforced HERE, outside the blocking call: a record
    still unfinished after ABANDON_SECONDS is written off and the loop moves on. The thread stays
    stuck, so once enough of them leak the pool is replaced to restore capacity.

    Rejected records are written to the manifest too (status='rejected'), because a filter you
    cannot audit is a filter you cannot trust: the reason column is what tells you whether the
    thresholds are throwing away good colour images. `rejected_dir` additionally keeps the pixels
    for a visual QA pass -- worth using on a sample before committing to a large scrape.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if rejected_dir is not None:
        rejected_dir.mkdir(parents=True, exist_ok=True)
    seen, index, already_accepted = loadManifest(manifest_path)
    is_new_manifest: bool = not manifest_path.exists()
    counts: Dict[str, int] = {"saved": already_accepted, "skipped_seen": 0,
                              "failed": 0, "rejected": 0, "abandoned": 0,
                              "host_skipped": 0,
                              "socket_closed": 0,
                              "attempted": 0}
    if already_accepted:
        print(f"wznowienie: {already_accepted} obrazow juz zapisanych, "
              f"brakuje {max(target_accepted - already_accepted, 0)}")
    if already_accepted >= target_accepted:
        return counts

    pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=workers)
    leaked: int = 0
    with open(manifest_path, "a", newline="") as manifest_file, \
            tqdm(total=target_accepted, initial=already_accepted,
                 desc=f"{category} accepted") as progress:
        writer: csv.writer = csv.writer(manifest_file)
        if is_new_manifest:
            writer.writerow(["filename", "index", "category", "record_id", "image_url",
                             "query", "width", "height", "status", "reason"])

        in_flight: Dict[Future, float] = {}
        host_of: Dict[Future, str] = {}
        handle_of: Dict[Future, Dict[str, object]] = {}
        strikes: Dict[str, int] = {}
        blocked_hosts: Set[str] = set()
        exhausted: bool = False
        last_heartbeat: float = time.monotonic()
        while counts["saved"] < target_accepted and (in_flight or not exhausted):
            # Keep the pool fed but bounded: an unbounded submit would buffer every record's
            # decoded image in memory long before the target is reached.
            while not exhausted and len(in_flight) < workers * 2:
                record: Optional[Dict[str, str]] = next(records, None)
                if record is None:
                    exhausted = True
                    break
                if record["record_id"] in seen:
                    counts["skipped_seen"] += 1
                    continue
                host: str = urlparse(record["image_url"]).netloc
                # One SaaS host (zetcom.group serves many museums) can own a long contiguous run
                # of the record stream. When it stops answering, abandoning record after record
                # to the same dead host burns the whole record budget for nothing.
                if host in blocked_hosts:
                    counts["host_skipped"] += 1
                    seen.add(record["record_id"])
                    continue
                if max_attempts and counts["attempted"] >= max_attempts:
                    tqdm.write(f"limit prob pobrania ({max_attempts}) wyczerpany")
                    exhausted = True
                    break
                seen.add(record["record_id"])
                counts["attempted"] += 1
                conn_handle: Dict[str, object] = {}
                future: Future = pool.submit(fetchAndFilter, record, conn_handle)
                in_flight[future] = time.monotonic()
                host_of[future] = host
                handle_of[future] = conn_handle

            if not in_flight:
                break
            finished, _ = futures.wait(list(in_flight), timeout=POLL_SECONDS,
                                       return_when=futures.FIRST_COMPLETED)

            now: float = time.monotonic()
            if now - last_heartbeat > HEARTBEAT_SECONDS:
                oldest: float = now - min(in_flight.values()) if in_flight else 0.0
                tqdm.write(f"[hb] w locie={len(in_flight)} gotowe={len(finished)} "
                           f"najstarszy={oldest:.0f}s pominiete={counts['skipped_seen']} "
                           f"porzucone={counts['abandoned']} odrzucone={counts['rejected']} "
                           f"bledy={counts['failed']} host_skip={counts['host_skipped']} "
                           f"zablokowane_hosty={len(blocked_hosts)} wyczerpane={exhausted}")
                last_heartbeat = now
            for future in [f for f, started in in_flight.items()
                           if f not in finished and now - started > ABANDON_SECONDS]:
                cancelled: bool = future.cancel()     # true only if it never started running
                del in_flight[future]
                counts["abandoned"] += 1
                # Tearing the socket down makes the blocked read raise, so the worker unwinds and
                # its thread and descriptor return to the pool. Without this every abandoned
                # record costs one thread and one fd for the rest of the process.
                if closeHandle(handle_of.pop(future, {})):
                    counts["socket_closed"] += 1
                elif not cancelled:
                    leaked += 1                      # ran, but had no socket we could reach
                strikeHost(host_of.pop(future, ""), strikes, blocked_hosts)
            if leaked >= workers:
                # Every worker is stuck on a dribbling server; a fresh pool restores throughput.
                tqdm.write(f"{leaked} zablokowanych watkow -- wymiana puli")
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=workers)
                leaked = 0

            for future in finished:
                del in_flight[future]
                finished_host: str = host_of.pop(future, "")
                handle_of.pop(future, None)
                result: Dict[str, object] = future.result()
                record = result["record"]                    # type: ignore[assignment]
                bgr = result["image"]                        # type: ignore[assignment]
                if result["status"] == "failed":
                    counts["failed"] += 1
                    # A host whose reads time out FAILS rather than stalls, so counting only
                    # abandonments left the breaker permanently unarmed: the scrape spent an hour
                    # timing out 16-at-a-time against one dead SaaS host, at 0.5 failures/s.
                    strikeHost(finished_host, strikes, blocked_hosts)
                    continue
                strikes.pop(finished_host, None)             # a host that answers is forgiven

                height, width = bgr.shape[:2]
                if result["status"] == "rejected":
                    counts["rejected"] += 1
                    rejected_name: str = ""
                    if rejected_dir is not None:
                        rejected_name = f"{category}_rej{counts['rejected']}.jpg"
                        cv2.imwrite(str(rejected_dir / rejected_name), bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    writer.writerow([rejected_name, "", category, record["record_id"],
                                     record["image_url"], query, width, height,
                                     "rejected", result["reason"]])
                    manifest_file.flush()
                    continue

                filename: str = f"{category}_n{index}.jpg"
                cv2.imwrite(str(out_dir / filename), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                writer.writerow([filename, index, category, record["record_id"],
                                 record["image_url"], query, width, height,
                                 "accepted", result["reason"]])
                manifest_file.flush()            # a long scrape must survive being interrupted
                index += 1
                counts["saved"] += 1
                progress.update(1)
    pool.shutdown(wait=False)                    # never join: a stuck worker would hang exit
    return counts


def readQueryFile(filename: str) -> Dict[str, List[str]]:
    query: Dict[str, List[str]] = {"q": [], "qf": []}
    with open(filename, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or "," not in line:
                continue
            query_type, query_value = line.split(",", 1)
            if query_type in query:
                query[query_type].append(query_value)
    return query


def parseArgs() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_path", type=str, help="plik z zapytaniami (q,... / qf,...)")
    parser.add_argument("category", type=str, help="kategoria, uzywana w nazwach plikow")
    parser.add_argument("number_of_images", type=int,
                        help="ile obrazow ZAAKCEPTOWANYCH zapisac (nie: ile rekordow pobrac)")
    parser.add_argument("--theme", type=str, default="")
    parser.add_argument("--workers", type=int, default=8, help="rownolegle pobrania")
    parser.add_argument("--max_attempts", type=int, default=0,
                        help="limit PROB POBRANIA (nie przejrzanych rekordow); 0 = 10x "
                             "number_of_images. Rekordy pominiete -- juz widziane albo z hosta "
                             "na czarnej liscie -- nic nie kosztuja i nie licza sie do limitu")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="domyslnie ./data/scraped/<category>")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="domyslnie <out_dir>/../manifest_<category>.csv")
    parser.add_argument("--rejected_dir", type=Path, default=None,
                        help="zapisuj odrzucone obrazy tutaj (do wizualnej kontroli filtra)")
    parser.add_argument("--wskey", type=str, default=os.environ.get("EUROPEANA_WSKEY", ""),
                        help="klucz API; domyslnie ze zmiennej EUROPEANA_WSKEY")
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parseArgs()
    if not args.wskey:
        raise SystemExit("Brak klucza API: ustaw EUROPEANA_WSKEY albo podaj --wskey")

    # `kill -USR1 <pid>` dumps every thread's stack to stderr. A scrape that stalls is otherwise
    # opaque -- the process looks alive, sockets look open, and py-spy needs root to attach.
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    # Catches a peer that goes completely silent. It does NOT catch the case that actually
    # stalled this scrape -- a server dribbling one byte per read keeps resetting every socket
    # timeout there is -- which is why the real bound lives in downloadRecords' abandon logic.
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    if not installConnectionRecorder():
        print('UWAGA: nie udalo sie podpiac pod urllib3 -- porzucone watki beda wyciekac')

    out_dir: Path = args.out_dir or Path("./data/scraped") / args.category
    manifest: Path = args.manifest or out_dir.parent / f"manifest_{args.category}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    query_dict: Dict[str, List[str]] = readQueryFile(args.query_path)
    query: str = " ".join(query_dict["q"]) or "*"
    # The stream limit only bounds paging; the real budget is download ATTEMPTS below. Counting
    # scanned records was a design error: on `industrial` 65% of the stream pointed at ten
    # blacklisted hosts, and skipping those -- which costs nothing -- ate 49k of the 88k budget,
    # so the category would have stopped at 89% of target with the API nowhere near exhausted.
    max_attempts: int = args.max_attempts or args.number_of_images * 10
    records: Iterator[Dict[str, str]] = streamRecords(
        query=query, qf=query_dict["qf"], theme=args.theme,
        wskey=args.wskey, limit=max_attempts * 20)

    counts: Dict[str, int] = downloadRecords(records, out_dir, manifest, args.category, query,
                                             target_accepted=args.number_of_images,
                                             workers=args.workers,
                                             max_attempts=max_attempts,
                                             rejected_dir=args.rejected_dir)
    examined: int = counts["rejected"] + counts["failed"]
    print(f"zapisano_lacznie={counts['saved']} (w tym z poprzednich uruchomien) "
          f"odrzucono={counts['rejected']} bledy={counts['failed']} "
          f"porzucone={counts['abandoned']} (gniazd zamknietych "
          f"{counts['socket_closed']}) host_skip={counts['host_skipped']} "
          f"juz_znane={counts['skipped_seen']} "
          f"przejrzano_teraz={examined + counts['saved']}")
    print(f"-> {out_dir}\n-> {manifest}")

    # Abandoned workers are still blocked in a socket read and are not daemon threads, so a
    # normal return would hang in the interpreter's thread join. Everything is already flushed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
