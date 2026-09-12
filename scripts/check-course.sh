#!/usr/bin/env bash
#
# Course integrity check. This is what CI runs, and what you should run after
# editing an exercise. It asserts three things:
#
#   1. Every reference solution passes. A solution that does not is worse than
#      no solution.
#   2. Every exercise STARTER fails -- either to import or to pass. An
#      exercise that already passes teaches nothing, and is the most common way
#      for one of these to rot as numpy and torch change underneath it.
#   3. Every starter that fails at *import* time says so in its header comment,
#      so a learner is never staring at a traceback the course did not warn
#      them about -- and, conversely, no header promises an import error that
#      the starter does not produce.
#
# Usage: scripts/check-course.sh [prefix]
#   With a prefix of an exercise id (e.g. "04_" for a chapter, "04_02" for one
#   exercise), only the matching exercises and solutions are checked. Useful
#   while writing a chapter.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FILTER="${1:-}"

readonly RED=$'\033[31m' GREEN=$'\033[32m' BOLD=$'\033[1m' RESET=$'\033[0m'

failures=0
note() { printf '  %sFAIL%s %s\n' "${RED}" "${RESET}" "$*"; failures=$((failures + 1)); }
ok() { printf '  %s ok %s %s\n' "${GREEN}" "${RESET}" "$*"; }

pytest_() { (cd "${ROOT}" && uv run --quiet pytest "$@"); }

# Directories of exercises (and their solutions), filtered.
exercise_dirs() {
  local dir chapter name id
  while IFS= read -r dir; do
    chapter="$(basename "$(dirname "${dir}")")"
    name="$(basename "${dir}")"
    id="${chapter%%_*}_${name}"
    [[ -z "${FILTER}" || "${id}" == "${FILTER}"* ]] || continue
    printf '%s\t%s\n' "${id}" "${dir}"
  done < <(find "${ROOT}/exercises" -mindepth 2 -maxdepth 2 -type d | sort)
}

printf '%s==> solutions must pass%s\n' "${BOLD}" "${RESET}"
solution_files=()
while IFS=$'\t' read -r id dir; do
  solution_files+=("${ROOT}/solutions/${dir#"${ROOT}"/exercises/}/exercise.py")
done < <(exercise_dirs)
log="$(mktemp)"
if pytest_ "${solution_files[@]}" -q --tb=short >"${log}" 2>&1; then
  ok "all solutions pass"
else
  grep -E '^(FAILED|ERROR)' "${log}" | sed 's/^/    /'
  tail -3 "${log}" | sed 's/^/    /'
  note "some solutions fail their own tests"
fi

printf '\n%s==> exercise starters must NOT pass%s\n' "${BOLD}" "${RESET}"
# One pytest session over all starters. A starter counts as failing if any of
# its tests failed or if it could not be collected (import-time error). The
# -rfE report lists exactly those two kinds of line, one per test or file.
starter_files=()
while IFS=$'\t' read -r id dir; do
  starter_files+=("${dir}/exercise.py")
done < <(exercise_dirs)
report="$(mktemp)"
pytest_ "${starter_files[@]}" -q --tb=no -rfE --continue-on-collection-errors \
  >"${report}" 2>&1 || true

passing=()
uncollectable=()
while IFS=$'\t' read -r id dir; do
  rel="${dir#"${ROOT}"/}/exercise.py"
  if grep -qE "^ERROR ${rel}" "${report}"; then
    uncollectable+=("${id}")
  elif ! grep -qE "^FAILED ${rel}" "${report}"; then
    passing+=("${id}")
  fi
done < <(exercise_dirs)

if [[ ${#passing[@]} -eq 0 ]]; then
  ok "every starter fails, as it should"
else
  for id in "${passing[@]}"; do
    note "starter ${id} already passes -- there is nothing to do in it"
  done
fi

printf '\n%s==> starters that do not import must say so%s\n' "${BOLD}" "${RESET}"
if [[ ${#uncollectable[@]} -eq 0 ]]; then
  ok "no starter needs a note"
else
  for id in "${uncollectable[@]}"; do
    chapter_number="${id%%_*}"
    rest="${id#*_}"
    dir="$(find "${ROOT}/exercises/${chapter_number}"_* -maxdepth 1 -type d \
      -name "${rest}" 2>/dev/null | head -1)"
    if [[ -n "${dir}" ]] && grep -qiE '^#  NOTE.*(import time|import error|does not import)' "${dir}/exercise.py"; then
      ok "${id} (documented)"
    else
      note "${id} fails at import time and has no NOTE in its header comment"
    fi
  done
fi

printf '\n%s==> starters that say they do not import must not import%s\n' "${BOLD}" "${RESET}"
lying=0
while IFS=$'\t' read -r id dir; do
  grep -qiE '^#  NOTE.*(import time|import error|does not import)' "${dir}/exercise.py" || continue
  if [[ " ${uncollectable[*]-} " != *" ${id} "* ]]; then
    note "${id} says it starts as an import error, but it imports"
    lying=$((lying + 1))
  fi
done < <(exercise_dirs)
if [[ ${lying} -eq 0 ]]; then
  ok "every NOTE is truthful"
fi
rm -f "${report}" "${log}"

printf '\n%s==> counting%s\n' "${BOLD}" "${RESET}"
if [[ -n "${FILTER}" ]]; then
  printf '  (tree-wide count skipped: checking only %s*)\n' "${FILTER}"
fi
count="$(find "${ROOT}/exercises" -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')"
solution_count="$(find "${ROOT}/solutions" -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')"
printf '  %s exercises, %s solutions\n' "${count}" "${solution_count}"
if [[ -z "${FILTER}" && "${count}" != "${solution_count}" ]]; then
  note "every exercise needs a solution"
fi
missing=0
while IFS=$'\t' read -r id dir; do
  [[ -f "${ROOT}/solutions/${dir#"${ROOT}"/exercises/}/exercise.py" ]] || {
    note "${id} has no solution file"; missing=1; }
done < <(exercise_dirs)

printf '\n'
if [[ ${failures} -eq 0 ]]; then
  printf '%s%sthe course is consistent%s\n' "${GREEN}" "${BOLD}" "${RESET}"
  exit 0
fi
printf '%s%s%d problem(s)%s\n' "${RED}" "${BOLD}" "${failures}" "${RESET}"
exit 1
