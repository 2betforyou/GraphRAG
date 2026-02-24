#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
조항 단위 GraphRAG 그래프 생성기 (2025-10 교차 법령 참조 확장 버전)
-------------------------------------------------------------------
입력: 법령 구조 JSON 디렉토리 (--json_dir)
출력: 그래프 JSON (--out_json) + (선택) PyVis HTML (--out_html)

변경점(교차 참조 지원):
  • "법령명 + 제n조" 패턴을 파싱해 명시적 교차 법령 참조 우선 연결
  • "제n조"만 있을 때는 전 법령 범위로 단일 매칭 시 교차 연결, 다중이면 현재 법령으로 폴백
  • "이 법", "같은 법"은 현재 법령으로 해석
  • (law, 제n조)와 제n조→노드목록 역색인으로 탐색 효율 개선
"""

import os, json, argparse, re
import networkx as nx
from tqdm import tqdm
from utils import extract_refs, infer_relation   # 기존 유틸 함수 사용

# --------------------------------------------------
# 유틸: 법령명 정규화
# --------------------------------------------------
def normalize_law_name(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"\s+", "", name)  # 공백 제거
    # 필요 시 아래에 추가 정규화 규칙(괄호/특수기호 제거 등) 확장 가능
    s = s.replace("-", "-").replace("–", "-")
    return s

# --------------------------------------------------
# 유틸: 교차 법령 참조 파서
#   - 반환: [{"law": "개인정보보호법", "article": "21"}, ...] 형태
#   - "이법", "같은법" → current_law 로 매핑
# --------------------------------------------------
CROSS_LAW_PATTERN = re.compile(
    r"(?P<law>[가-힣A-Za-z0-9\(\)·\-\s]{2,}?)\s*제\s*(?P<num>\d+)\s*조"
)

def parse_cross_law_refs(text: str, current_law: str):
    results = []
    for m in CROSS_LAW_PATTERN.finditer(text):
        law_raw = m.group("law").strip()
        num = m.group("num").strip()

        # '이 법', '같은 법' → 현재 법령
        if re.fullmatch(r"(이\s*법|같은\s*법)", law_raw):
            law_norm = current_law
        else:
            # 문맥 잡음(예: '제목', '별표' 등) 제거용 빠른 필터링
            # 확장 가능: 금지 단어 사전
            if any(tok in law_raw for tok in ["제목", "별표", "부칙", "참조"]):
                continue
            law_norm = normalize_law_name(law_raw)

        results.append({"law": law_norm, "article": num})
    return results

# --------------------------------------------------
# 조항 수집
# --------------------------------------------------
def collect_articles(node, prefix):
    """트리 구조 JSON에서 '조' 단위 텍스트 병합"""
    results = []
    if node.get("label") == "조":
        title = node.get("title", "")
        text = node.get("text", "")
        parts = [text]
        for ch in node.get("children", []):
            if isinstance(ch, dict):
                parts.append(ch.get("text", ""))
                for subch in ch.get("children", []):
                    parts.append(subch.get("text", ""))
        full_text = "\n".join([p for p in parts if p and p.strip()])
        aid = f"{prefix}_{title.replace(' ', '')}"
        results.append((aid, title, full_text, prefix))
    for child in node.get("children", []):
        results.extend(collect_articles(child, prefix))
    return results

# --------------------------------------------------
# 동일 조항 병합
# --------------------------------------------------
def merge_duplicate_articles(articles):
    """같은 제n조가 여러 항/문장으로 분리된 경우 병합"""
    merged = {}
    for aid, title, text, law in articles:
        # 제21조, 제21조의2 등 변형을 최대한 보존하면서 기본 키를 구성
        m = re.match(r"\s*제\s*(\d+)\s*조(의\s*\d+)?", title)
        if m:
            base = "제" + m.group(1) + "조"
            if m.group(2):
                # '의2' 같은 꼬리도 보존
                base += m.group(2).replace(" ", "")
        else:
            # 백업: 원제목에서 '제\d+조'만 추출
            base_find = re.search(r"제\s*\d+\s*조(의\s*\d+)?", title)
            base = base_find.group(0).replace(" ", "") if base_find else title.strip()

        key = f"{law}_{base}"
        merged.setdefault(key, {"law": law, "title": base, "texts": []})["texts"].append(text)

    merged_articles = []
    for k, v in merged.items():
        merged_articles.append((k, v["title"], "\n".join(v["texts"]), v["law"]))
    return merged_articles

# --------------------------------------------------
# 메인
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="법령 구조 JSON 디렉토리 경로")
    ap.add_argument("--out_json", required=True, help="출력 그래프 JSON 경로")
    ap.add_argument("--out_html", required=False, default=None, help="(선택) PyVis 시각화 HTML 경로")
    args = ap.parse_args()

    G = nx.DiGraph()

    # 파일 목록 로드
    files = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]

    # 법령별 표시명(alias) 준비: 파일명(prefix) ↔ 표시명(title)
    # (데이터 루트에 "title"이 있다면 사용, 없으면 prefix 사용)
    law_display_by_prefix = {}
    law_prefix_by_normalized_name = {}

    # 1차 패스: 노드 수집 및 병합, 동시에 법령명 사전 구축
    all_merged_by_file = {}
    for fn in tqdm(files, desc="Collecting & merging articles"):
        prefix = os.path.splitext(fn)[0]
        path = os.path.join(args.json_dir, fn)
        data = json.load(open(path, encoding="utf-8"))

        # 법령 표시명 추출(가능하면)
        law_display = data.get("title") or prefix
        law_display_by_prefix[prefix] = law_display

        # 정규화된 이름으로 prefix 매핑(교차 참조 시 역찾기 용도)
        norm_name = normalize_law_name(law_display)
        law_prefix_by_normalized_name[norm_name] = prefix

        articles = collect_articles(data, prefix)
        merged_articles = merge_duplicate_articles(articles)
        all_merged_by_file[prefix] = merged_articles

    # 2차 패스: 그래프 노드 추가
    for prefix, merged_articles in all_merged_by_file.items():
        for aid, title, text, law in merged_articles:
            G.add_node(aid, title=title, law=law, text=text)

    # -------- 역색인 구축: (law, 제n조) → node_id,  제n조 → [node_ids] --------
    by_law_and_article = {}   # {(law_prefix, '제n조' 혹은 '제n조의m'): node_id}
    by_article_only = {}      # {'제n조' 혹은 '제n조의m': [node_ids across laws]}

    def article_key_from_title(title: str) -> str:
        # '제21조', '제21조의2' 모두 포괄
        m = re.match(r"\s*(제\s*\d+\s*조(\s*의\s*\d+)?)", title)
        return m.group(1).replace(" ", "") if m else title.replace(" ", "")

    for nid in G.nodes:
        node = G.nodes[nid]
        law = node.get("law")
        akey = article_key_from_title(node.get("title", ""))
        by_law_and_article[(law, akey)] = nid
        by_article_only.setdefault(akey, []).append(nid)

    # 3차 패스: 참조(엣지) 추가 — 교차 법령 지원
    for prefix, merged_articles in tqdm(all_merged_by_file.items(), desc="Linking references"):
        for aid, title, text, law in merged_articles:
            # (A) 명시적 교차 참조: "법령명 + 제n조" 우선 연결
            cross_refs = parse_cross_law_refs(text, current_law=prefix)

            # (B) 단순 조번호 참조(동일 함수 유지): utils.extract_refs → [n, ...]
            simple_refs = extract_refs(text)  # "제n조" 패턴에서 숫자만 추출한다고 가정

            # 우선: 명시적 교차 참조
            for r in cross_refs:
                law_name_norm = r["law"]
                num = r["article"]
                akey = f"제{num}조"
                # (law명 → prefix) 변환
                target_prefix = law_prefix_by_normalized_name.get(law_name_norm)
                if not target_prefix:
                    # 법령명이 텍스트에 있지만 사전에 없으면 스킵
                    continue
                nid = by_law_and_article.get((target_prefix, akey))
                if nid and nid != aid:
                    rel = infer_relation(text[:300])
                    G.add_edge(aid, nid, relation=rel)

            # 다음: 단순 "제n조" 참조
            for num in simple_refs:
                akey = f"제{num}조"
                candidates = by_article_only.get(akey, [])
                if not candidates:
                    continue

                rel = infer_relation(text[:300])

                # 후보가 1개면 → 교차든 동일이든 그쪽으로 바로 연결
                if len(candidates) == 1:
                    nid = candidates[0]
                    if nid != aid:
                        G.add_edge(aid, nid, relation=rel)
                    continue

                # 후보가 여러 개면 → 현재 법령과 정확히 일치하는 노드가 있으면 그쪽으로
                this_law_nid = by_law_and_article.get((prefix, akey))
                if this_law_nid and this_law_nid in candidates and this_law_nid != aid:
                    G.add_edge(aid, this_law_nid, relation=rel)
                else:
                    # 현재 법령에 없고 후보가 여러 개인 경우:
                    # (1) 제목이 '제n조의m' 등으로 더 구체화된 유일 후보가 있는지 확인
                    # (단순화: 동일 akey 기준이므로 보통은 다수 그대로 남음)
                    # 안전하게는 연결 생략(중복·오연결 방지). 필요 시 전략 변경 가능.
                    pass

    # --------------------------------------------------
    # JSON 저장
    # --------------------------------------------------
    out = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"src": u, "dst": v, **G.edges[u, v]} for u, v in G.edges]
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------
    # PyVis 시각화 (선택)
    # --------------------------------------------------
    if args.out_html:
        from pyvis.network import Network
        net = Network(height="900px", width="100%", directed=True)
        color_map = {"적용": "#6baed6", "예외": "#fd8d3c", "벌칙": "#de2d26",
                     "면제": "#31a354", "참조": "#999999"}

        for n in G.nodes:
            node = G.nodes[n]
            law = node.get("law", "unknown")
            # 표시용: 파일 prefix 대신 실제 표시명 사용
            law_disp = law
            # law_display_by_prefix가 있으면 치환
            if law in law_display_by_prefix:
                law_disp = law_display_by_prefix[law]
            title = node.get("title", "")
            label = f"{law_disp} {title}".strip()
            tooltip = node.get("text", "")[:400].replace("\n", " ")
            net.add_node(n, label=label, title=tooltip)

        for u, v in G.edges:
            rel = G.edges[u, v].get("relation", "참조")
            net.add_edge(u, v, label=rel, color=color_map.get(rel, "#999999"))

        # 브라우저 연동 없이 파일 저장
        out_html_dir = os.path.dirname(args.out_html)
        if out_html_dir:
            os.makedirs(out_html_dir, exist_ok=True)
        net.write_html(args.out_html)

    print(f"Graph build complete — Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

if __name__ == "__main__":
    main()