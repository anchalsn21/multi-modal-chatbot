"""
Generate Greenfield University Campus Map as SVG.
Legend sits below the map — no overlap, no clipping.
"""

svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="820" viewBox="0 0 960 820">
  <defs>
    <style>
      .title       { font-family: Georgia, serif; font-size: 20px; font-weight: bold; fill: white; }
      .zone-label  { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold;
                     fill: rgba(0,0,0,0.30); letter-spacing: 2px; }
      .mk-circle   { stroke: #fff; stroke-width: 2; }
      .mk-letter   { font-family: Arial Black, Arial, sans-serif; font-size: 11px; font-weight: 900;
                     fill: white; text-anchor: middle; dominant-baseline: central; }
      .mk-name     { font-family: Arial, sans-serif; font-size: 8.5px; font-weight: bold;
                     fill: #1a1a2e; text-anchor: middle; }
      .mk-sub      { font-family: Arial, sans-serif; font-size: 7.5px; fill: #444;
                     text-anchor: middle; }
      .gate-label  { font-family: Arial, sans-serif; font-size: 8px; font-weight: bold;
                     fill: #5d4037; text-anchor: middle; }
      .road        { fill: none; stroke: #bfb49e; stroke-width: 7; stroke-linecap: round; }
      .road-inner  { fill: none; stroke: #f0e8d8; stroke-width: 3.5; stroke-linecap: round; }
      .leg-title   { font-family: Arial, sans-serif; font-size: 10px; font-weight: bold; fill: #1a1a2e; }
      .leg-text    { font-family: Arial, sans-serif; font-size: 9px; fill: #333; }
      .compass-txt { font-family: Arial, sans-serif; font-size: 10px; font-weight: bold;
                     fill: #1a1a2e; text-anchor: middle; }
    </style>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="rgba(0,0,0,0.22)" />
    </filter>
    <filter id="lsh" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="1.5" flood-color="rgba(0,0,0,0.15)" />
    </filter>
  </defs>

  <!-- ── Page background ── -->
  <rect width="960" height="820" fill="#ede9e0" />

  <!-- ══════════════════════════════════════════════════════════
       TITLE BAR
  ══════════════════════════════════════════════════════════ -->
  <rect x="30" y="10" width="900" height="36" rx="8" fill="#1a1a2e" filter="url(#sh)" />
  <text x="480" y="34" text-anchor="middle" class="title">Greenfield University — Campus Map</text>

  <!-- ══════════════════════════════════════════════════════════
       CAMPUS BOUNDARY  (shifted right to leave space on left)
  ══════════════════════════════════════════════════════════ -->
  <rect x="80" y="55" width="790" height="590" rx="16" ry="16"
        fill="#d8d3c8" stroke="#7a6e5e" stroke-width="2.5" filter="url(#sh)" />

  <!-- ── Zone fills ── -->
  <!-- North -->
  <rect x="84"  y="58"  width="782" height="178" rx="10" fill="#cce5f7" stroke="#85b8d4" stroke-width="1.2" opacity="0.92" />
  <!-- South -->
  <rect x="84"  y="432" width="782" height="208" rx="10" fill="#c5e8c6" stroke="#78c87a" stroke-width="1.2" opacity="0.92" />
  <!-- West -->
  <rect x="84"  y="236" width="196" height="196" rx="8"  fill="#e0bce8" stroke="#b57bc8" stroke-width="1.2" opacity="0.92" />
  <!-- East -->
  <rect x="598" y="236" width="278" height="196" rx="8"  fill="#ffdcaa" stroke="#ffb040" stroke-width="1.2" opacity="0.92" />
  <!-- Central -->
  <rect x="280" y="236" width="318" height="196" rx="8"  fill="#fffcb8" stroke="#f0d820" stroke-width="1.2" opacity="0.92" />

  <!-- ── Roads ── -->
  <!-- Horizontal spine -->
  <line x1="84"  y1="334" x2="872" y2="334" class="road" />
  <line x1="84"  y1="334" x2="872" y2="334" class="road-inner" />
  <!-- Vertical spine -->
  <line x1="476" y1="58"  x2="476" y2="640" class="road" />
  <line x1="476" y1="58"  x2="476" y2="640" class="road-inner" />
  <!-- NW diagonal -->
  <path d="M 280 236 Q 230 196 190 162" class="road" />
  <path d="M 280 236 Q 230 196 190 162" class="road-inner" />
  <!-- NE diagonal -->
  <path d="M 598 236 Q 648 196 698 162" class="road" />
  <path d="M 598 236 Q 648 196 698 162" class="road-inner" />
  <!-- SW diagonal -->
  <path d="M 280 432 Q 230 478 190 510" class="road" />
  <path d="M 280 432 Q 230 478 190 510" class="road-inner" />
  <!-- SE diagonal -->
  <path d="M 598 432 Q 648 478 698 510" class="road" />
  <path d="M 598 432 Q 648 478 698 510" class="road-inner" />
  <!-- Inner ring road -->
  <ellipse cx="476" cy="334" rx="155" ry="78"
           fill="none" stroke="#bfb49e" stroke-width="6" opacity="0.55" />
  <ellipse cx="476" cy="334" rx="155" ry="78"
           fill="none" stroke="#f0e8d8" stroke-width="3" opacity="0.75" />

  <!-- ── Zone labels ── -->
  <text x="476" y="92"  text-anchor="middle" class="zone-label">NORTH CAMPUS</text>
  <text x="476" y="618" text-anchor="middle" class="zone-label">SOUTH CAMPUS</text>
  <text x="182" y="340" text-anchor="middle" class="zone-label" transform="rotate(-90,182,340)">WEST CAMPUS</text>
  <text x="688" y="340" text-anchor="middle" class="zone-label" transform="rotate(90,688,340)">EAST CAMPUS</text>
  <text x="439" y="258" text-anchor="middle" class="zone-label">CENTRAL CAMPUS</text>

  <!-- ── Campus gates ── -->
  <rect x="456" y="53"  width="40" height="10" rx="3" fill="#8d6e63" stroke="#5d4037" stroke-width="1.2" />
  <text x="476" y="78"  class="gate-label">NORTH GATE</text>

  <rect x="456" y="631" width="40" height="10" rx="3" fill="#5d4037" stroke="#3e2723" stroke-width="1.8" />
  <polygon points="476,641 466,654 486,654" fill="#5d4037" />
  <text x="476" y="668" class="gate-label" font-size="9px" font-weight="bold">MAIN GATE</text>

  <rect x="870" y="325" width="10" height="20" rx="3" fill="#8d6e63" stroke="#5d4037" stroke-width="1.2" />
  <text x="892" y="337" class="gate-label" text-anchor="start">EAST GATE</text>

  <rect x="80"  y="325" width="10" height="20" rx="3" fill="#8d6e63" stroke="#5d4037" stroke-width="1.2" />
  <text x="70"  y="337" class="gate-label" text-anchor="end">WEST GATE</text>

  <!-- South Gate (separate from Main Gate) -->
  <rect x="456" y="631" width="40" height="10" rx="3" fill="#8d6e63" stroke="#5d4037" stroke-width="1.2" opacity="0.5"/>
  <text x="380" y="648" class="gate-label">SOUTH GATE</text>

  <!-- ══════════════════════════════════════════════════════════
       MAP MARKERS
  ══════════════════════════════════════════════════════════ -->

  <!-- Helper: each marker = circle + letter + label bg + 2-line label -->

  <!-- ─── NORTH CAMPUS (blue #1565c0) ─── -->

  <!-- A = Main Library -->
  <circle cx="200" cy="128" r="15" class="mk-circle" fill="#1565c0" filter="url(#lsh)" />
  <text x="200" y="129" class="mk-letter">A</text>
  <rect x="162" y="147" width="77" height="24" rx="3" fill="white" opacity="0.85" />
  <text x="200" y="157" class="mk-name">Main Library</text>
  <text x="200" y="167" class="mk-sub">(Hartley Bldg)</text>

  <!-- B = 24-Hour Study Suite -->
  <circle cx="370" cy="115" r="15" class="mk-circle" fill="#1565c0" filter="url(#lsh)" />
  <text x="370" y="116" class="mk-letter">B</text>
  <rect x="330" y="134" width="80" height="24" rx="3" fill="white" opacity="0.85" />
  <text x="370" y="144" class="mk-name">24hr Study Suite</text>
  <text x="370" y="154" class="mk-sub">(Murray Bldg)</text>

  <!-- F = International Student Office -->
  <circle cx="570" cy="128" r="15" class="mk-circle" fill="#1565c0" filter="url(#lsh)" />
  <text x="570" y="129" class="mk-letter">F</text>
  <rect x="528" y="147" width="84" height="24" rx="3" fill="white" opacity="0.85" />
  <text x="570" y="157" class="mk-name">Intl Student Office</text>
  <text x="570" y="167" class="mk-sub">(Global House)</text>

  <!-- G = Computer Science Dept -->
  <circle cx="760" cy="118" r="15" class="mk-circle" fill="#1565c0" filter="url(#lsh)" />
  <text x="760" y="119" class="mk-letter">G</text>
  <rect x="720" y="137" width="80" height="24" rx="3" fill="white" opacity="0.85" />
  <text x="760" y="147" class="mk-name">CS Department</text>
  <text x="760" y="157" class="mk-sub">(Zepler Bldg)</text>

  <!-- ─── CENTRAL CAMPUS (amber #f57f17) ─── -->

  <!-- C = Main Cafeteria -->
  <circle cx="360" cy="308" r="15" class="mk-circle" fill="#e65c00" filter="url(#lsh)" />
  <text x="360" y="309" class="mk-letter">C</text>
  <rect x="318" y="327" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="360" y="337" class="mk-name">Main Cafeteria</text>
  <text x="360" y="347" class="mk-sub">(Cavendish Bldg)</text>

  <!-- D = Student Union -->
  <circle cx="476" cy="344" r="15" class="mk-circle" fill="#e65c00" filter="url(#lsh)" />
  <text x="476" y="345" class="mk-letter">D</text>
  <rect x="434" y="363" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="476" y="373" class="mk-name">Student Union</text>
  <text x="476" y="383" class="mk-sub">(Union House)</text>

  <!-- E = Administration -->
  <circle cx="560" cy="290" r="15" class="mk-circle" fill="#e65c00" filter="url(#lsh)" />
  <text x="560" y="291" class="mk-letter">E</text>
  <rect x="520" y="309" width="80" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="560" y="319" class="mk-name">Admin Office</text>
  <text x="560" y="329" class="mk-sub">(Chancellor Bldg)</text>

  <!-- ─── SOUTH CAMPUS (green #2e7d32) ─── -->

  <!-- H = Medical Centre -->
  <circle cx="220" cy="490" r="15" class="mk-circle" fill="#2e7d32" filter="url(#lsh)" />
  <text x="220" y="491" class="mk-letter">H</text>
  <rect x="178" y="509" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="220" y="519" class="mk-name">Medical Centre</text>
  <text x="220" y="529" class="mk-sub">(Wellbeing Hub)</text>

  <!-- I = Psychology Dept -->
  <circle cx="476" cy="508" r="15" class="mk-circle" fill="#2e7d32" filter="url(#lsh)" />
  <text x="476" y="509" class="mk-letter">I</text>
  <rect x="432" y="527" width="88" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="476" y="537" class="mk-name">Psychology Dept</text>
  <text x="476" y="547" class="mk-sub">(Meadows Bldg)</text>

  <!-- J = Counselling Centre -->
  <circle cx="720" cy="490" r="15" class="mk-circle" fill="#2e7d32" filter="url(#lsh)" />
  <text x="720" y="491" class="mk-letter">J</text>
  <rect x="676" y="509" width="88" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="720" y="519" class="mk-name">Counselling Centre</text>
  <text x="720" y="529" class="mk-sub">(Meadows Bldg)</text>

  <!-- ─── EAST CAMPUS (orange #e65100) ─── -->

  <!-- K = Sports Centre -->
  <circle cx="668" cy="270" r="15" class="mk-circle" fill="#bf360c" filter="url(#lsh)" />
  <text x="668" y="271" class="mk-letter">K</text>
  <rect x="626" y="289" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="668" y="299" class="mk-name">Sports Centre</text>
  <text x="668" y="309" class="mk-sub">(Athletico Cplx)</text>

  <!-- L = Innovation Lab -->
  <circle cx="790" cy="324" r="15" class="mk-circle" fill="#bf360c" filter="url(#lsh)" />
  <text x="790" y="325" class="mk-letter">L</text>
  <rect x="748" y="343" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="790" y="353" class="mk-name">Innovation Lab</text>
  <text x="790" y="363" class="mk-sub">(Tech Hub)</text>

  <!-- M = Engineering Faculty -->
  <circle cx="730" cy="400" r="15" class="mk-circle" fill="#bf360c" filter="url(#lsh)" />
  <text x="730" y="401" class="mk-letter">M</text>
  <rect x="685" y="419" width="90" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="730" y="429" class="mk-name">Engineering Faculty</text>
  <text x="730" y="439" class="mk-sub">(Brunel Tower)</text>

  <!-- ─── WEST CAMPUS (purple #6a1b9a) ─── -->

  <!-- N = Career Services -->
  <circle cx="174" cy="268" r="15" class="mk-circle" fill="#6a1b9a" filter="url(#lsh)" />
  <text x="174" y="269" class="mk-letter">N</text>
  <rect x="132" y="287" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="174" y="297" class="mk-name">Career Services</text>
  <text x="174" y="307" class="mk-sub">(Futures Bldg)</text>

  <!-- O = PG Common Room -->
  <circle cx="130" cy="348" r="15" class="mk-circle" fill="#6a1b9a" filter="url(#lsh)" />
  <text x="130" y="349" class="mk-letter">O</text>
  <rect x="88"  y="367" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="130" y="377" class="mk-name">PG Common Room</text>
  <text x="130" y="387" class="mk-sub">(Futures, 2F)</text>

  <!-- P = Arts & Design -->
  <circle cx="174" cy="416" r="15" class="mk-circle" fill="#6a1b9a" filter="url(#lsh)" />
  <text x="174" y="417" class="mk-letter">P</text>
  <rect x="132" y="435" width="84" height="24" rx="3" fill="white" opacity="0.88" />
  <text x="174" y="445" class="mk-name">Arts &amp; Design</text>
  <text x="174" y="455" class="mk-sub">(Renwick Bldg)</text>

  <!-- ── Decorative greenery ── -->
  <circle cx="290" cy="180" r="9"  fill="#a5d6a7" opacity="0.65" />
  <circle cx="305" cy="174" r="7"  fill="#81c784" opacity="0.65" />
  <circle cx="278" cy="190" r="6"  fill="#a5d6a7" opacity="0.55" />
  <circle cx="520" cy="176" r="9"  fill="#a5d6a7" opacity="0.65" />
  <circle cx="535" cy="170" r="7"  fill="#81c784" opacity="0.65" />
  <ellipse cx="476" cy="575" rx="52" ry="22" fill="#a5d6a7" opacity="0.45" />
  <circle cx="456" cy="570" r="10" fill="#66bb6a" opacity="0.55" />
  <circle cx="476" cy="564" r="12" fill="#a5d6a7" opacity="0.55" />
  <circle cx="496" cy="572" r="9"  fill="#66bb6a" opacity="0.55" />

  <!-- Fountain -->
  <circle cx="476" cy="300" r="11" fill="#81d4fa" stroke="#0288d1" stroke-width="1.5" opacity="0.75" />
  <circle cx="476" cy="300" r="5"  fill="#e1f5fe" opacity="0.9" />
  <text x="476" y="319" text-anchor="middle" font-family="Arial" font-size="8px" fill="#0277bd">Fountain</text>

  <!-- ══════════════════════════════════════════════════════════
       COMPASS ROSE  (top-right, inside campus area)
  ══════════════════════════════════════════════════════════ -->
  <g transform="translate(848, 96)">
    <circle cx="0" cy="0" r="26" fill="white" stroke="#666" stroke-width="1.2" opacity="0.92" />
    <polygon points="0,-20 -6,-4 6,-4"  fill="#c0392b" />
    <polygon points="0,20  -6,4  6,4"   fill="#777" />
    <line x1="-20" y1="0" x2="20" y2="0" stroke="#777" stroke-width="1.2" />
    <text x="0"   y="-23" class="compass-txt" font-size="10px" fill="#c0392b">N</text>
    <text x="0"   y="31"  class="compass-txt" font-size="9px">S</text>
    <text x="25"  y="4"   class="compass-txt" font-size="9px" text-anchor="start">E</text>
    <text x="-25" y="4"   class="compass-txt" font-size="9px" text-anchor="end">W</text>
  </g>

  <!-- ══════════════════════════════════════════════════════════
       LEGEND  — below the campus map, 4-column grid
  ══════════════════════════════════════════════════════════ -->
  <rect x="30" y="660" width="900" height="148" rx="8" fill="white" stroke="#ccc"
        stroke-width="1" opacity="0.95" filter="url(#lsh)" />

  <!-- Section heading -->
  <text x="480" y="681" text-anchor="middle" class="leg-title" font-size="11px">MAP LEGEND — Zones A to P</text>
  <line x1="40" y1="686" x2="920" y2="686" stroke="#ddd" stroke-width="1" />

  <!-- Column 1 (x=50) — A B C D -->
  <circle cx="52"  cy="700" r="6" fill="#1565c0" /><text x="63" y="704"  class="leg-text">A — Main Library (Hartley Bldg)</text>
  <circle cx="52"  cy="718" r="6" fill="#1565c0" /><text x="63" y="722"  class="leg-text">B — 24-Hour Study Suite (Murray)</text>
  <circle cx="52"  cy="736" r="6" fill="#e65c00" /><text x="63" y="740"  class="leg-text">C — Main Cafeteria (Cavendish)</text>
  <circle cx="52"  cy="754" r="6" fill="#e65c00" /><text x="63" y="758"  class="leg-text">D — Student Union (Union House)</text>

  <!-- Column 2 (x=280) — E F G H -->
  <circle cx="282" cy="700" r="6" fill="#e65c00" /><text x="293" y="704"  class="leg-text">E — Admin Office (Chancellor Bldg)</text>
  <circle cx="282" cy="718" r="6" fill="#1565c0" /><text x="293" y="722"  class="leg-text">F — Intl Student Office (Global House)</text>
  <circle cx="282" cy="736" r="6" fill="#1565c0" /><text x="293" y="740"  class="leg-text">G — CS Department (Zepler Bldg)</text>
  <circle cx="282" cy="754" r="6" fill="#2e7d32" /><text x="293" y="758"  class="leg-text">H — Medical Centre (Wellbeing Hub)</text>

  <!-- Column 3 (x=530) — I J K L -->
  <circle cx="532" cy="700" r="6" fill="#2e7d32" /><text x="543" y="704"  class="leg-text">I  — Psychology Dept (Meadows Bldg)</text>
  <circle cx="532" cy="718" r="6" fill="#2e7d32" /><text x="543" y="722"  class="leg-text">J  — Counselling Centre (Meadows)</text>
  <circle cx="532" cy="736" r="6" fill="#bf360c" /><text x="543" y="740"  class="leg-text">K — Sports Centre (Athletico Cplx)</text>
  <circle cx="532" cy="754" r="6" fill="#bf360c" /><text x="543" y="758"  class="leg-text">L  — Innovation Lab (Tech Hub)</text>

  <!-- Column 4 (x=760) — M N O P -->
  <circle cx="762" cy="700" r="6" fill="#bf360c" /><text x="773" y="704"  class="leg-text">M — Engineering (Brunel Tower)</text>
  <circle cx="762" cy="718" r="6" fill="#6a1b9a" /><text x="773" y="722"  class="leg-text">N — Career Services (Futures Bldg)</text>
  <circle cx="762" cy="736" r="6" fill="#6a1b9a" /><text x="773" y="740"  class="leg-text">O — PG Common Room (Futures, 2F)</text>
  <circle cx="762" cy="754" r="6" fill="#6a1b9a" /><text x="773" y="758"  class="leg-text">P  — Arts &amp; Design Studio (Renwick)</text>

  <!-- Zone colour key row -->
  <text x="480" y="782" text-anchor="middle" font-family="Arial" font-size="8.5px" fill="#666">
    Zone colours:
  </text>
  <rect x="530" y="772" width="14" height="10" rx="2" fill="#cce5f7" stroke="#85b8d4" stroke-width="0.8" />
  <text x="548" y="781" font-family="Arial" font-size="8px" fill="#444">North</text>
  <rect x="590" y="772" width="14" height="10" rx="2" fill="#c5e8c6" stroke="#78c87a" stroke-width="0.8" />
  <text x="608" y="781" font-family="Arial" font-size="8px" fill="#444">South</text>
  <rect x="650" y="772" width="14" height="10" rx="2" fill="#e0bce8" stroke="#b57bc8" stroke-width="0.8" />
  <text x="668" y="781" font-family="Arial" font-size="8px" fill="#444">West</text>
  <rect x="705" y="772" width="14" height="10" rx="2" fill="#ffdcaa" stroke="#ffb040" stroke-width="0.8" />
  <text x="723" y="781" font-family="Arial" font-size="8px" fill="#444">East</text>
  <rect x="760" y="772" width="14" height="10" rx="2" fill="#fffcb8" stroke="#f0d820" stroke-width="0.8" />
  <text x="778" y="781" font-family="Arial" font-size="8px" fill="#444">Central</text>

</svg>'''

output_path = r"e:\Assignment\BSBI AI\MMC 2\campus_assistant\frontend\public\campus_map.svg"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"SVG campus map saved to: {output_path}")
print(f"File size: {len(svg_content)} bytes")
