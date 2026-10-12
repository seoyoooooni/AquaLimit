"""실제 검증 지역(California Aqueduct MP 195~215) 공개 자료 내려받기.

1) DWR TRE Altamira InSAR 포인트 자료(5GB zip)에서 HTTP range 요청으로 필요한 타일만 스트리밍해 범위 안 점만 남긴다.
2) DWR 정기 지하수위 관측정(stations)·측정값(2014~)을 CKAN SQL API로 받는다.
3) State Water Project 반복구간(repayment reach) 선형과 정밀측량점 위치를 받는다.

출력: real/raw/ (원본 잘라낸 것, git 제외), real/data/ (가공 자료)
"""
import csv, json, os, struct, subprocess, sys, urllib.parse, zlib
import remote_zip as rz

BOX = dict(lat0=35.50, lat1=35.78, lon0=-119.92, lon1=-119.58)
TILES = [21, 22, 23]  # 포인트 자료는 위도 내림차순으로 타일이 나뉘어 있음 (21: 35.79°~, 23: ~35.40°)
RAW = "raw"; os.makedirs(RAW, exist_ok=True)

def inbox(la, lo):
    return BOX["lat0"] <= la <= BOX["lat1"] and BOX["lon0"] <= lo <= BOX["lon1"]

def fetch_insar():
    es = {e[0]: e for e in rz.entries()}
    for i in TILES:
        dst = f"{RAW}/insar_tile{i}.csv"
        if os.path.exists(dst): continue
        _, meth, csz, usz, loff = es[f"verticaldisplacementpointdata/CALIFORNIA_DWR_{i}_VERT.csv"]
        h, _ = rz.rng(loff, loff + 29); nl, el = struct.unpack("<HH", h[26:30]); s = loff + 30 + nl + el
        p = subprocess.Popen(["curl", "-sSL", "-r", f"{s}-{s+csz-1}", rz.U], stdout=subprocess.PIPE)
        d = zlib.decompressobj(-15); buf = b""; first = True
        with open(dst + ".part", "w") as out:
            while True:
                b = p.stdout.read(4 << 20)
                if not b: break
                buf += d.decompress(b); lines = buf.split(b"\n"); buf = lines.pop()
                for L in lines:
                    if first:
                        first = False; out.write(L.decode() + "\n"); continue
                    a = L.split(b",", 3)
                    try:
                        if inbox(float(a[1]), float(a[2])): out.write(L.decode() + "\n")
                    except ValueError: pass
        os.rename(dst + ".part", dst); print("insar tile", i, "done")

def ckan_sql(sql):
    u = "https://data.cnra.ca.gov/api/3/action/datastore_search_sql?sql=" + urllib.parse.quote(sql)
    return json.loads(subprocess.run(["curl", "-sS", "--max-time", "90", u], capture_output=True).stdout)["result"]["records"]

def fetch_wells():
    st = f"{RAW}/gwl_stations_all.csv"
    if not os.path.exists(st):
        subprocess.run(["curl", "-sSL", "-o", st, "https://data.cnra.ca.gov/dataset/dd9b15f5-6d08-4d8c-bace-37dc761a9c08/resource/af157380-fb42-4abf-b72a-6f9f98868077/download/stations.csv"], check=True)
    rows = [r for r in csv.DictReader(open(st, encoding="utf-8", errors="replace"))
            if r["latitude"] and inbox(float(r["latitude"]), float(r["longitude"]))]
    rid = "bfa9f262-24a1-45bd-8dc8-138bc8107266"; ms = []
    ids = [r["site_code"] for r in rows]
    for k in range(0, len(ids), 40):
        ch = ",".join(f"'{i}'" for i in ids[k:k+40])
        ms += ckan_sql(f'SELECT site_code,msmt_date,gwe,gse_gwe,wlm_qa_desc FROM "{rid}" WHERE site_code IN ({ch}) AND msmt_date >= \'2014-01-01\'')
    have = {m["site_code"] for m in ms if m["gwe"] not in (None, "")}
    keep = ["site_code","stn_id","well_name","latitude","longitude","gse","well_depth","well_use","well_type","monitoring_program"]
    with open("data/wells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep); w.writeheader()
        for r in rows:
            if r["site_code"] in have: w.writerow({k: r[k] for k in keep})
    with open("data/gwl_measurements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_code","msmt_date","gwe","gse_gwe","wlm_qa_desc"]); w.writeheader()
        for m in sorted(ms, key=lambda m: (m["site_code"], m["msmt_date"])):
            if m["gwe"] not in (None, ""): w.writerow({k: m[k] for k in w.fieldnames})
    print("wells with data", len(have), "measurements", sum(1 for m in ms if m["gwe"] not in (None, "")))

def fetch_aqueduct():
    for name, item in [("swp_reaches", "7ab987dcb3314e799cd998defe171dd1"), ("swp_precise_surveys", "f3d04b1dd46748fe927e77fc17021889")]:
        dst = f"{RAW}/{name}.geojson"
        if not os.path.exists(dst):
            subprocess.run(["curl", "-sSL", "-o", dst, f"https://gis.data.cnra.ca.gov/api/download/v1/items/{item}/geojson?layers=0"], check=True)
    d = json.load(open(f"{RAW}/swp_reaches.geojson"))
    feats = [f for f in d["features"] if f["properties"]["Reach_Number"] in ("CA-R9", "CA-R10A", "CA-R11B")]
    json.dump({"type": "FeatureCollection", "features": feats}, open("data/aqueduct_mp185_218.geojson", "w"))
    s = json.load(open(f"{RAW}/swp_precise_surveys.geojson"))
    pts = [f for f in s["features"] if inbox(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])]
    json.dump({"type": "FeatureCollection", "features": pts}, open("data/precise_survey_marks.geojson", "w"))
    print("aqueduct reaches", len(feats), "survey marks", len(pts))

def fetch_oil():
    """CalGEM 유전 경계: 석유 생산 침하를 지하수 침하와 구분하기 위함."""
    from shapely.geometry import shape, box
    dst = f"{RAW}/oil_fields.geojson"
    if not os.path.exists(dst):
        subprocess.run(["curl", "-sSL", "-o", dst, "https://gis-cnra.hub.arcgis.com/api/download/v1/items/624ef97c019549548f1e9396cbc39e77/geojson?layers=0"], check=True)
    B = box(BOX["lon0"], BOX["lat0"], BOX["lon1"], BOX["lat1"])
    d = json.load(open(dst))
    sel = [f for f in d["features"] if f["geometry"] and shape(f["geometry"]).intersects(B)]
    json.dump({"type": "FeatureCollection", "features": sel}, open("data/oil_fields_in_box.geojson", "w"))
    print("oil fields", len(sel))

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    what = sys.argv[1:] or ["insar", "wells", "aqueduct", "oil"]
    if "insar" in what: fetch_insar()
    if "wells" in what: fetch_wells()
    if "aqueduct" in what: fetch_aqueduct()
    if "oil" in what: fetch_oil()
