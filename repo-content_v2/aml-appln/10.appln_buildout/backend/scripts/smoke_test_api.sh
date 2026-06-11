#!/usr/bin/env bash
# Smoke-test every backend endpoint listed in api_documentation.md.
# Usage:  bash scripts/smoke_test_api.sh [BASE_URL]
#   BASE_URL defaults to http://localhost:8010
#
# Output: one line per endpoint:
#   PASS | FAIL <http_code> | SKIP <reason>   METHOD path
#
# Exit code 0 if all PASS, 1 if any FAIL.

set -u
BASE="${1:-http://localhost:8010}"
ENT="${SMOKE_ENTITY_ID:-SYN_2291b518}"
CASE="${SMOKE_CASE_ID:-DEMO_0001}"
ALERT="${SMOKE_ALERT_ID:-DEMO_0001}"
SOP="${SMOKE_SOP_ID:-SOP-STRUCTURING-01}"

pass=0
fail=0
declare -a failures=()

check() {
  local name="$1" expected="$2" method="$3" path="$4" body="${5:-}"
  local code
  if [ "$method" = "GET" ]; then
    code=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" "$BASE$path")
  else
    code=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" -X "$method" \
             -H "Content-Type: application/json" -d "$body" "$BASE$path")
  fi
  if [ "$code" = "$expected" ]; then
    printf "  PASS   %-6s %s\n" "$method" "$path"
    pass=$((pass+1))
  else
    printf "  FAIL %s %-6s %s\n" "$code" "$method" "$path"
    fail=$((fail+1))
    failures+=("$method $path -> $code")
  fi
}

probe_body() {
  # Returns "PASS" if the JSON body does NOT contain {"error": ...} at top level.
  local name="$1" method="$2" path="$3" body="${4:-}"
  local out
  if [ "$method" = "GET" ]; then
    out=$(/usr/bin/curl -s "$BASE$path")
  else
    out=$(/usr/bin/curl -s -X "$method" -H "Content-Type: application/json" \
             -d "$body" "$BASE$path")
  fi
  # Strip {"value": ...} envelope if present
  local payload
  payload=$(printf '%s' "$out" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    if isinstance(d, dict) and set(d.keys()) == {'value'}:
        d = d['value']
    # Flag only when 'error' carries a non-empty, non-null value at the top level.
    # Successful endpoints often carry 'error': null in their schema.
    err = d.get('error') if isinstance(d, dict) else None
    if err:
        print('ERR:', err)
    else:
        print('OK')
except Exception as e:
    print('PARSE_ERR:', e)
" 2>&1)
  if [[ "$payload" == OK* ]]; then
    printf "  PASS   %-6s %s\n" "$method" "$path"
    pass=$((pass+1))
  else
    printf "  FAIL   %-6s %s   (%s)\n" "$method" "$path" "$payload"
    fail=$((fail+1))
    failures+=("$method $path -> body: $payload")
  fi
}

echo "==================== AML Backend smoke test ===================="
echo "BASE  = $BASE"
echo "ENT   = $ENT"
echo "CASE  = $CASE"
echo

echo "[1] Health & system"
check  ""  200  GET   /api/health
probe_body system_config         GET   /api/system/config
probe_body system_components     GET   /api/system/components

echo
echo "[2] Alerts"
check  ""  200  GET   /api/alerts
check  ""  200  GET   "/api/alerts?status=open&limit=5"
check  ""  200  GET   /api/alerts/stats
probe_body alerts_get_post POST /api/alerts/get '{"alert_id":"'"$ALERT"'"}'
check  ""  422  GET   "/api/alerts/$ALERT"

echo
echo "[3] Entities"
check  ""  200  GET   /api/entities
check  ""  200  GET   "/api/entities?risk_rating=low&limit=5"
probe_body entities_get_post                POST /api/entities/get                '{"entity_id":"'"$ENT"'"}'
probe_body entities_transactions_post       POST /api/entities/transactions       '{"entity_id":"'"$ENT"'","limit":3}'
probe_body entities_behavioral_post         POST /api/entities/behavioral_summary '{"entity_id":"'"$ENT"'"}'
probe_body entities_risk_post               POST /api/entities/risk_score         '{"entity_id":"'"$ENT"'"}'
probe_body entities_network_post            POST /api/entities/network            '{"entity_id":"'"$ENT"'","depth":2}'
probe_body entities_timeline_post           POST /api/entities/timeline           '{"entity_id":"'"$ENT"'"}'

echo
echo "[4] Network"
probe_body network_global   GET  /api/network/global
probe_body network_patterns GET  /api/network/patterns
probe_body network_path     POST /api/network/path     '{"source":"'"$ENT"'","target":"AI Trading LLC"}'

echo
echo "[5] Policy / SOP / Sanctions"
probe_body policy_search  POST /api/policy/search    '{"typology":"structuring","q":"10000","k":4}'
probe_body policy_sources GET  /api/policy/sources
probe_body sops_list      GET  /api/sops
probe_body sops_get       POST /api/sops/get         '{"sop_id":"'"$SOP"'"}'
probe_body sanctions      POST /api/sanctions/screen '{"name":"ACME Trading LLC","min_score":0.55}'

echo
echo "[6] Analytics"
for p in overview typology_distribution risk_heatmap timeline channel_mix top_counterparties aux_usage agent_performance profile; do
  probe_body analytics_$p GET /api/analytics/$p
done

echo
echo "[7] Skills (require NIM at :8088)"
NIM_UP=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" http://localhost:8088/v1/models)
if [ "$NIM_UP" != "200" ]; then
  echo "  SKIP   NIM at :8088 is not reachable (got $NIM_UP)"
else
  probe_body skill_behavioral POST /api/skills/behavioral '{"passage":"## KYC profile\nentity_id: X\n\n## Transactions\ndate,amount,currency,counterparty,channel,notes\n2026-03-01,9500,USD,Branch ATM,cash,"}'
  probe_body skill_numeric    POST /api/skills/numeric    '{"passage":"deposits 9500, 9800, 9750","question":"Sum the deposits"}'
  probe_body skill_citation   POST /api/skills/citation   '{"passage":"FinCEN advisory: sub-threshold cash deposits constitute structuring under 31 USC 5324.","question":"What does this say about structuring?"}'
  probe_body skill_statutory  POST /api/skills/statutory  '{"statute":"31 USC 5324(a)(3)","fact_pattern":"8 cash deposits of 9500 USD in 8 days","question":"Does this fall under the statute?"}'
fi

echo
echo "[8] Investigation"
# Try to find an existing trace file to test against; the endpoint returns an
# error body for cases that haven't been investigated yet.
TRACES_DIR="${NAT_AML_DATA_DIR:-$PWD/data}/traces"
EXISTING_CASE=""
if [ -d "$TRACES_DIR" ]; then
  EXISTING_CASE=$(ls "$TRACES_DIR"/*.json 2>/dev/null | head -1 | xargs -I{} basename {} .json)
fi
if [ -z "$EXISTING_CASE" ]; then
  echo "  SKIP   POST   /api/investigation/get (no traces in $TRACES_DIR — run an investigation first)"
else
  probe_body investigation_get POST /api/investigation/get "{\"case_id\":\"$EXISTING_CASE\"}"
fi

echo
echo "[9] Demo eval (read-only — gated by NAT_AML_EVAL_TOKEN when set)"
probe_body demo_eval_runs   GET  /api/demo/eval/runs
probe_body demo_eval        POST /api/demo/eval                  '{}'
probe_body demo_eval_cases  POST /api/demo/eval/cases            '{"limit":5}'
if [ -z "$EXISTING_CASE" ]; then
  echo "  SKIP   POST   /api/demo/eval/case/{case_id} (no traces present)"
else
  probe_body demo_eval_case POST "/api/demo/eval/case/$EXISTING_CASE" "{\"case_id\":\"$EXISTING_CASE\",\"include_full_tool_outputs\":false}"
fi
# Compare requires two distinct snapshot directories. Discover and pick two.
RUNS_JSON=$(/usr/bin/curl -s "$BASE/api/demo/eval/runs")
RUN_A=$(printf '%s' "$RUNS_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin); v=d.get('value',d)
items=[x['name'] for x in v.get('items',[])]
custom=[n for n in items if 'custom' in n]
print(custom[0] if custom else (items[0] if items else ''))
")
RUN_B=$(printf '%s' "$RUNS_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin); v=d.get('value',d)
items=[x['name'] for x in v.get('items',[])]
base=[n for n in items if 'base' in n]
print(base[0] if base else (items[1] if len(items)>1 else ''))
")
if [ -z "$RUN_A" ] || [ -z "$RUN_B" ] || [ "$RUN_A" = "$RUN_B" ]; then
  echo "  SKIP   POST   /api/demo/eval/compare (need two distinct snapshot dirs; have RUN_A=$RUN_A RUN_B=$RUN_B)"
else
  probe_body demo_eval_compare POST /api/demo/eval/compare "{\"run_a\":\"$RUN_A\",\"run_b\":\"$RUN_B\"}"
fi
probe_body demo_model_cmp   POST /api/demo/eval/model_comparison '{"report":"latest"}'

echo
echo "[10] Frontend shape contract (fields the UI accesses must exist)"
shape_check() {
  local name="$1" method="$2" path="$3" body="$4" must_have="$5"
  local out payload
  if [ "$method" = "GET" ]; then
    out=$(/usr/bin/curl -s "$BASE$path")
  else
    out=$(/usr/bin/curl -s -X "$method" -H "Content-Type: application/json" -d "$body" "$BASE$path")
  fi
  payload=$(printf '%s' "$out" | MUST_HAVE="$must_have" python3 -c "
import json, sys, os
d = json.loads(sys.stdin.read())
if isinstance(d, dict) and set(d.keys()) == {'value'}: d = d['value']
required = os.environ['MUST_HAVE'].split(',')
missing = [r.strip() for r in required if r.strip() and r.strip() not in (d or {})]
print('MISSING:' + ','.join(missing) if missing else 'OK')
" 2>&1)
  if [ "$payload" = "OK" ]; then
    printf "  PASS   %-6s %s  (has %s)\n" "$method" "$path" "$must_have"
    pass=$((pass+1))
  else
    printf "  FAIL   %-6s %s  (%s)\n" "$method" "$path" "$payload"
    fail=$((fail+1))
    failures+=("$method $path shape: $payload")
  fi
}

shape_check policy_sources GET  /api/policy/sources                         ''                              sources
shape_check sops_list      GET  /api/sops                                    ''                              sops
shape_check sanctions      POST /api/sanctions/screen                        '{"name":"ACME LLC"}'           'name,items'
shape_check policy_search  POST /api/policy/search                           '{"typology":"structuring","k":1}' 'typology,items'
shape_check entity_get     POST /api/entities/get                            '{"entity_id":"'"$ENT"'"}'      'kyc,n_tx_total,channel_mix,related_alerts'
shape_check alert_get      POST /api/alerts/get                              '{"alert_id":"'"$ALERT"'"}'    'alert,status,kyc_snippet'
shape_check overview       GET  /api/analytics/overview                      ''                              'n_alerts_total,n_entities,n_transactions'
shape_check typology_dist  GET  /api/analytics/typology_distribution         ''                              'seeded,from_traces'
shape_check agent_perf     GET  /api/analytics/agent_performance             ''                              'per_typology,n_traces'
shape_check model_cmp      POST /api/demo/eval/model_comparison              '{"report":"latest"}'           'headline_metrics,confusion,endpoints'

echo
echo "==================== Summary ===================="
echo "PASS: $pass"
echo "FAIL: $fail"
if [ "$fail" -gt 0 ]; then
  echo
  echo "Failures:"
  for f in "${failures[@]}"; do echo "  - $f"; done
  exit 1
fi
exit 0
