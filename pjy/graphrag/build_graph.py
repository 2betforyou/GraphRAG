#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
조항 단위 GraphRAG 그래프 생성기 (2025-10 안정화 버전)
---------------------------------------------------------
입력: 법령 구조 JSON 디렉토리 (--json_dir)
출력: 그래프 JSON (--out_json) + (선택) PyVis HTML (--out_html)
특징:
  • 각 법령의 '제n조' 단위로 병합하여 노드 안정화
  • law 속성 유지 (KeyError 방지)
  • 관계 추출 (적용, 예외, 벌칙 등) 자동 반영
"""

import os, json, argparse, re
import networkx as nx
from tqdm import tqdm
from utils import extract_refs, infer_relation


# --------------------------------------------------
# 조항 수집 함수
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
        full_text = "\n".join([p for p in parts if p.strip()])
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
        base = re.sub(r'제\s*(\d+)\s*조.*', r'제\1조', title)
        key = f"{law}_{base.strip()}"
        if key not in merged:
            merged[key] = {"law": law, "title": base, "texts": [text]}
        else:
            merged[key]["texts"].append(text)

    merged_articles = []
    for k, v in merged.items():
        merged_articles.append(
            (k, v["title"], "\n".join(v["texts"]), v["law"])
        )
    return merged_articles


# --------------------------------------------------
# 메인 함수
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="법령 구조 JSON 디렉토리 경로")
    ap.add_argument("--out_json", required=True, help="출력 그래프 JSON 경로")
    ap.add_argument("--out_html", required=False, default=None, help="(선택) PyVis 시각화 HTML 경로")
    args = ap.parse_args()

    G = nx.DiGraph()
    files = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]

    for fn in tqdm(files, desc="Building law graph"):
        prefix = os.path.splitext(fn)[0]
        path = os.path.join(args.json_dir, fn)
        data = json.load(open(path, encoding="utf-8"))

        # 1) 조항 수집
        articles = collect_articles(data, prefix)

        # 2) 조항 병합
        merged_articles = merge_duplicate_articles(articles)

        # 3) 노드 추가
        for aid, title, text, law in merged_articles:
            G.add_node(aid, title=title, law=law, text=text)

        # 4) 참조 관계 추가
        for aid, title, text, law in merged_articles:
            refs = extract_refs(text)
            rel = infer_relation(text[:300])            # 본문 앞 300자만 보고 관계 라벨을 추정함. (효율성/정확성 균형 이라는데, 이거 빼도 상관 없을듯)
            for r in refs:
                tid = f"{law}_제{r}조"
                # 노드 탐색 (fallback 포함)
                target = None
                for nid in G.nodes:
                    if f"제{r}조" in nid and law in nid:
                        target = nid
                        break
                if target and target != aid:
                    G.add_edge(aid, target, relation=rel)

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
            title = node.get("title", "")
            label = f"{law} {title}".strip()
            tooltip = node.get("text", "")[:400].replace("\n", " ")
            net.add_node(n, label=label, title=tooltip)

        for u, v in G.edges:
            rel = G.edges[u, v].get("relation", "참조")
            net.add_edge(u, v, label=rel, color=color_map.get(rel, "#999999"))

        # net.show(args.out_html)

    print(f"Graph build complete — Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")


if __name__ == "__main__":
    main()