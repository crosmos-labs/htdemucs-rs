/**
 * Dump the aligner requests the Fretboard pipeline sends for each song, with no stem paths, so the
 * A/B harness can replay them on other stems. Run from /Users/divyansh/guitar-tabs (read-only use):
 *   npx vite-node -c scripts/song.config.ts /Users/divyansh/htdemucs-rs/research/round1/dump_requests.ts <out.json> <id>...
 * Per song: chords (the chords stage's request at the sounding shift, `chroma` to be filled in),
 * tab (the pinned part, = tabRequestFor) and, extra (not a pipeline gate), each vocal part of the
 * Songsterr transcription aligned on its own, to get a sung-pitch match on more than one song.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { chordsRequestFor, loadScore, notesRequest, tabRequestFor } from '/Users/divyansh/guitar-tabs/server/pipeline/build.ts';

const [out, ...ids] = process.argv.slice(2);
const recipeOf = (id: string) => {
  for (const d of ['data/songs', '.cache/eval/songs']) {
    const f = `${d}/${id}.recipe.json`;
    if (existsSync(f)) return JSON.parse(readFileSync(f, 'utf8'));
  }
  throw new Error(`no recipe for ${id}`);
};
const res: Record<string, unknown> = {};
for (const id of ids) {
  const r = recipeOf(id);
  const entry: Record<string, unknown> = { recordingSha256: r.recording?.sha256 ?? null, chordsShift: r.chords?.soundingShift ?? null };
  if (r.chords) entry.chords = await chordsRequestFor(r, 'HARMONIC');
  if (r.tab) {
    entry.tab = await tabRequestFor(r);
    entry.tabPart = { index: r.tab.partIndex, melody: r.tab.melody, name: r.tab.partName };
    const vocal = (r.tab.parts ?? []).filter((p: any) => p.vocal);
    const vreq: unknown[] = [];
    if (r.tab.source === 'songsterr') {
      for (const p of vocal.slice(0, 1)) {
        try {
          const { score } = await loadScore(r, p.index);
          vreq.push({ index: p.index, name: p.name, req: { track: id, ...notesRequest(score, r.tab.tempoScale) } });
        } catch (e) {
          vreq.push({ index: p.index, name: p.name, error: String(e) });
        }
      }
    }
    entry.vocalParts = vreq;
  }
  res[id] = entry;
  console.error(id, Object.keys(entry).join(','));
}
writeFileSync(out, JSON.stringify(res));
