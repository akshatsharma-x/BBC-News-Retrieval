"""Write data/qrels.json from the graded relevance assessments.

Assessment procedure (see report section 4.2):
  1. depth-20 pooling over the five evaluated runs (scripts/build_pool.py),
     shown to the assessor in doc-id order without system or rank labels;
  2. pool enrichment: an interactive keyword sweep per topic surfaced
     unpooled candidates, which were judged with the same criteria.
Grades: 2 = highly relevant (the article is about the topic),
        1 = partially relevant (the topic is a substantial secondary theme),
        0 = not relevant. Every pooled document not listed below is graded 0.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELEVANT = {
 "Q01": "bus-003:2 bus-025:2 bus-077:2 bus-083:2 bus-127:2 bus-131:1 bus-157:1 bus-164:2 bus-181:1 bus-192:2 bus-230:2 bus-266:2 bus-299:2 bus-314:2 bus-318:2 bus-324:2 bus-332:2 bus-358:2 bus-391:2 bus-399:2 bus-426:2 bus-444:2 bus-456:1",
 "Q02": "bus-002:1 bus-089:2 bus-097:2 bus-142:2 bus-180:2 bus-273:1 bus-344:2 bus-350:2 bus-374:2 bus-376:2 bus-429:1 bus-440:1 bus-488:1 bus-496:2 bus-328:1 bus-024:1",
 "Q03": "bus-171:2 bus-203:2 bus-220:2 bus-264:2 bus-279:2 bus-316:2 bus-319:2 bus-392:2 bus-421:2 bus-454:2",
 "Q04": "bus-228:2 bus-260:2 bus-261:2 bus-262:2 bus-311:1 bus-345:2 bus-346:2 bus-353:2 bus-354:2 bus-398:2 bus-411:2 bus-434:2 bus-471:2 bus-491:2 pol-017:2",
 "Q05": "bus-059:2 bus-087:2 bus-100:2 bus-114:2 bus-196:2 bus-246:2 bus-403:2 bus-465:2",
 "Q06": "ent-038:2 ent-070:1 ent-352:2 ent-354:2 ent-355:2 ent-357:2",
 "Q07": "ent-046:2 ent-049:2 ent-053:2 ent-065:2 ent-066:2 ent-072:2 ent-073:1 ent-074:2 ent-076:2 ent-083:2 ent-297:2 ent-340:2 ent-363:1 ent-364:2 ent-380:2 ent-384:2 ent-100:2 ent-060:2 ent-312:2",
 "Q08": "ent-112:2 ent-144:2 ent-163:1 ent-233:1 ent-234:2 ent-240:2 ent-242:1",
 "Q09": "ent-129:1 ent-249:2 ent-251:2 ent-253:2 ent-254:1 ent-256:2 ent-131:1 ent-236:1 ent-255:1",
 "Q10": "ent-109:2 ent-130:1 ent-143:2 ent-157:1 ent-159:2 ent-160:2",
 "Q11": "pol-049:2 pol-081:2 pol-141:2 pol-227:2 pol-299:2",
 "Q12": "pol-032:2 pol-066:2 pol-134:2 pol-137:2 pol-152:2 pol-169:2 pol-179:2 pol-381:2",
 "Q13": "pol-111:2 pol-157:1 pol-237:1 pol-267:1 pol-281:2 pol-288:2 pol-293:2 pol-325:2 pol-333:1",
 "Q14": "pol-014:2 pol-040:2 pol-070:2 pol-072:2 pol-097:1 pol-100:1 pol-160:2 pol-182:2 pol-192:1 pol-205:1 pol-216:1 pol-221:2 pol-224:2 pol-290:2 pol-368:2 pol-378:2 pol-384:2 pol-279:1",
 "Q15": "pol-013:2 pol-087:2 pol-106:2 pol-122:2 pol-193:1 pol-201:2 pol-334:1 pol-382:2 pol-371:1 pol-147:1 pol-411:1 pol-408:1",
 "Q16": "spo-016:2 spo-017:2 spo-023:2 spo-033:2 spo-034:1 spo-047:2 spo-048:2 spo-050:2 spo-051:2 spo-054:2 spo-079:2 spo-080:2 spo-081:2 spo-082:2 spo-089:2",
 "Q17": "spo-441:1 spo-445:1 spo-447:2 spo-450:1 spo-457:1 spo-467:2 spo-471:2 spo-475:2 spo-476:2 spo-483:2 spo-486:2 spo-487:2 spo-489:1 spo-490:1 spo-496:2 spo-503:2 spo-477:2 spo-446:2 spo-478:1 spo-479:1 spo-480:2 spo-488:2",
 "Q18": "spo-103:1 spo-104:1 spo-125:1 spo-139:1 spo-140:1 spo-159:2 spo-169:2 spo-172:2 spo-183:2 spo-218:2 spo-231:2 spo-278:1 spo-280:2",
 "Q19": "spo-072:2 pol-154:1 pol-186:1",
 "Q20": "spo-291:2 spo-292:2 spo-303:2 spo-312:2 spo-316:1 spo-320:2 spo-357:1 spo-395:2 spo-396:2 spo-401:2 spo-403:2 spo-413:1 spo-416:2 spo-355:2 spo-359:1 spo-304:2 spo-313:1 spo-307:2 spo-308:1 spo-310:2 spo-341:2 spo-356:1 spo-337:2 spo-348:2",
 "Q21": "tec-040:2 tec-047:1 tec-110:1 tec-127:1 tec-148:2 tec-172:1 tec-183:1 tec-260:1 tec-263:1 tec-300:1 tec-383:1 tec-198:1",
 "Q22": "tec-003:2 tec-007:1 tec-008:2 tec-026:2 tec-027:1 tec-034:2 tec-039:2 tec-055:2 tec-060:1 tec-077:2 tec-096:2 tec-117:2 tec-147:2 tec-177:2 tec-189:2 tec-197:2 tec-210:2 tec-214:2 tec-217:2 tec-227:2 tec-257:2 tec-270:1 tec-273:2 tec-281:2 tec-284:1 tec-316:1 tec-357:2 tec-362:2 tec-385:2 tec-390:2 tec-122:2 tec-272:2 tec-250:2 tec-290:2 tec-116:2 tec-083:1 tec-020:1 tec-132:2 tec-188:1 tec-358:1 tec-368:2",
 "Q23": "tec-025:2 tec-045:2 tec-084:2 tec-112:2 tec-153:2 tec-185:2 tec-238:2 tec-244:1 tec-247:2 tec-341:2 tec-345:2 tec-351:2 tec-359:1 tec-363:2",
 "Q24": "ent-148:1 ent-271:1 tec-011:2 tec-038:2 tec-075:2 tec-091:2 tec-092:2 tec-145:1 tec-156:2 tec-206:1 tec-241:2 tec-255:2 tec-285:1 tec-286:2 tec-370:2 tec-391:2",
 "Q25": "bus-079:1 bus-236:1 tec-027:2 tec-031:2 tec-049:2 tec-340:2 tec-073:2 tec-120:2 tec-164:1 tec-188:1 tec-217:2 tec-227:1 tec-281:1 tec-369:1 tec-397:1 tec-077:1",
}

# Unpooled candidates surfaced by the keyword sweep and judged NOT relevant.
SWEEP_NONRELEVANT = {
 "Q01": "bus-091", "Q02": "bus-152", "Q04": "tec-027 tec-106 pol-135 pol-191 tec-142",
 "Q09": "ent-122", "Q10": "ent-234", "Q11": "pol-041 pol-159 pol-326", "Q12": "spo-185",
 "Q14": "bus-458", "Q15": "pol-041 pol-082 pol-383", "Q17": "spo-460 spo-465 spo-497 spo-430",
 "Q20": "spo-297 spo-129 spo-115 spo-331 spo-339 spo-332 spo-326 spo-352 tec-028",
 "Q21": "tec-009", "Q22": "tec-091 tec-207", "Q24": "tec-083 tec-354", "Q25": "tec-055",
}


def main() -> None:
    pool = json.loads((ROOT / "data" / "pool.json").read_text())["pool"]
    qrels, swept = {}, {}
    for qid, pooled in pool.items():
        judged = {doc: 0 for doc in pooled}
        rel = dict((d, int(g)) for d, g in (x.split(":") for x in RELEVANT[qid].split()))
        swept[qid] = sorted(d for d in rel if d not in judged) + SWEEP_NONRELEVANT.get(qid, "").split()
        for doc in SWEEP_NONRELEVANT.get(qid, "").split():
            judged.setdefault(doc, 0)
        judged.update(rel)
        qrels[qid] = dict(sorted(judged.items()))
    (ROOT / "data" / "qrels.json").write_text(json.dumps(qrels, indent=1))
    n_rel = {q: sum(g > 0 for g in v.values()) for q, v in qrels.items()}
    n_swept_rel = sum(1 for q in pool for d, g in qrels[q].items() if g > 0 and d not in pool[q])
    print("judged pairs:", sum(len(v) for v in qrels.values()),
          "| relevant:", sum(n_rel.values()),
          "| highly relevant:", sum(g == 2 for v in qrels.values() for g in v.values()),
          "| relevant found only by keyword sweep:", n_swept_rel)
    print("relevant per query:", n_rel)


if __name__ == "__main__":
    main()
