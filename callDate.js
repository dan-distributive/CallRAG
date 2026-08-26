// Turbobridge recording filenames encode the call's date/time:
// <accountId>-<YYMMDDHHMMSS>.mp3, e.g. 11167358-260825155928.mp3 ->
// 2026-08-25 15:59:28. Verified against a real file's mtime
// (11167358-231201164133.mp3 -> 2023-12-01, matched the file's actual
// modification date). Returns null for filenames that don't match (e.g.
// test clips), so callers can fall back gracefully rather than fail.
function parseCallDate(filename) {
  const m = filename.match(/-(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.mp3$/i);
  if (!m) return null;
  const [, yy, mm, dd, hh, mi, ss] = m;
  return new Date(Date.UTC(2000 + Number(yy), Number(mm) - 1, Number(dd), Number(hh), Number(mi), Number(ss))).toISOString();
}

module.exports = { parseCallDate };
