#!/usr/bin/env node
// Regenerates tests/fixtures/sample.pdf — a hand-rolled, dependency-free
// 3-page PDF with known text on each page. Committed as a binary fixture so
// the Read-tool PDF path has deterministic coverage without pulling in a
// PDF-authoring dependency.
//
//   node tests/setup/make-pdf-fixture.mjs
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outPath = join(here, "..", "fixtures", "sample.pdf");

const PAGES = [
  "Alpha page one carrot elephant",
  "Beta page two mango dolphin",
  "Gamma page three walnut penguin",
];

/** Escape a string for a PDF literal string. */
const esc = (s) => s.replace(/([\\()])/g, "\\$1");

const objects = [];
/** Push an object body and return its 1-based object number. */
const add = (body) => {
  objects.push(body);
  return objects.length;
};

// Reserve 1=Catalog, 2=Pages, 3=Font; page/content objects follow.
add(""); // 1 catalog (filled below)
add(""); // 2 pages
add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"); // 3 font

const pageRefs = [];
for (const text of PAGES) {
  const stream = `BT /F1 24 Tf 72 700 Td (${esc(text)}) Tj ET`;
  const contentNum = add(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  const pageNum = add(
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ` +
      `/Resources << /Font << /F1 3 0 R >> >> /Contents ${contentNum} 0 R >>`,
  );
  pageRefs.push(pageNum);
}

objects[0] = "<< /Type /Catalog /Pages 2 0 R >>";
objects[1] =
  `<< /Type /Pages /Count ${pageRefs.length} ` +
  `/Kids [${pageRefs.map((n) => `${n} 0 R`).join(" ")}] >>`;

let pdf = "%PDF-1.4\n";
const offsets = [];
objects.forEach((body, i) => {
  offsets.push(pdf.length);
  pdf += `${i + 1} 0 obj\n${body}\nendobj\n`;
});

const xrefStart = pdf.length;
pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
for (const off of offsets) pdf += `${String(off).padStart(10, "0")} 00000 n \n`;
pdf +=
  `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\n` + `startxref\n${xrefStart}\n%%EOF\n`;

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, Buffer.from(pdf, "latin1"));
console.log(`wrote ${outPath} (${pdf.length} bytes, ${PAGES.length} pages)`);
