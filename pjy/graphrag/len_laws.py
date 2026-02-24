import json

data = json.load(open("../../dataset/laws/graphrag/law_graph.json", encoding="utf-8"))
cross_edges = []
for e in data["edges"]:
    src_law = e["src"].split("_")[0]
    dst_law = e["dst"].split("_")[0]
    if src_law != dst_law:
        cross_edges.append(e)

print(f"전체 엣지 수: {len(data['edges'])}")
print(f"교차 법령 엣지 수: {len(cross_edges)}")
print(f"비율: {len(cross_edges)/len(data['edges'])*100:.2f}%")
