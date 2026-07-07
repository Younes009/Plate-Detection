# Schéma d'architecture — code LaTeX (TikZ) pour Overleaf

Compiler en **pdfLaTeX** (moteur par défaut d'Overleaf). La classe
`standalone` produit un PDF cadré sur le schéma, réutilisable dans un
rapport via `\includegraphics`.

Ajustements rapides : espacements via `node distance` et les `right=1.5cm
of ...` ; polices via `font=`.

```latex
\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc}

\begin{document}
\begin{tikzpicture}[
  font=\small,
  node distance=1.4cm and 1.5cm,
  % ── styles ────────────────────────────────────────────────
  service/.style={draw=blue!50!black, thick, fill=blue!8, rounded corners=3pt,
                  align=center, minimum height=1.5cm, inner sep=6pt},
  mlserv/.style={draw=orange!70!black, thick, fill=orange!12, rounded corners=3pt,
                 align=center, minimum height=1.5cm, inner sep=6pt},
  store/.style={draw=gray!60!black, thick, fill=gray!12, cylinder,
                shape border rotate=90, aspect=0.15, align=center, inner sep=5pt},
  note/.style={draw=gray!50, fill=yellow!15, rounded corners=2pt, align=left,
               font=\scriptsize, inner sep=4pt},
  step/.style={circle, draw=black, fill=white, thick, font=\bfseries\scriptsize,
               inner sep=1.5pt, minimum size=4.5mm},
  cflow/.style={-{Stealth[length=3mm]}, very thick, blue!60!black},
  wflow/.style={-{Stealth[length=3mm]}, very thick, orange!80!black},
  mflow/.style={-{Stealth[length=3mm]}, very thick, red!70!black, dashed},
  lbl/.style={font=\scriptsize\itshape, align=center},
]

% ════════════════ FLUX CONTINU (haut) ════════════════
\node[store] (src1) {\textbf{SOURCE/}\\\texttt{continuous\_ingress/}\\images brutes\\\scriptsize(aucun label)};

\node[service, right=1.5cm of src1] (prod1)
  {\textbf{PRODUCER}\\Microservice 1\\Spark Core + Hadoop FS\\[2pt]
   \scriptsize mode \texttt{continuous}\\\scriptsize job \'EPH\'EM\`ERE\\\scriptsize (transf\`ere puis exit 0)};

\node[store, right=1.5cm of prod1] (dest1)
  {\textbf{DEST/}\\\texttt{continuous\_ingress/}};

\node[service, right=1.7cm of dest1, minimum width=4.6cm] (cons)
  {\textbf{CONSUMER \& PREDICTOR}\\Microservice 2 --- Spark Structured Streaming\\[2pt]
   \scriptsize \texttt{readStream.format("binaryFile")}\\
   \scriptsize trigger 10 s $\cdot$ \texttt{awaitTermination}\\
   \scriptsize service PERMANENT (seul du pipeline)};

\node[store, right=1.7cm of cons] (out)
  {\textbf{OUTPUT/}\\Parquet Snappy\\\scriptsize prediction $\cdot$ confidence\\\scriptsize bbox\_cx/cy/w/h};

\node[service, right=1.5cm of out, fill=green!8, draw=green!50!black] (dash)
  {\textbf{DASHBOARD}\\Streamlit\\[2pt]
   \scriptsize Onglet 1 : Pipeline Tracker\\\scriptsize tra\c{c}abilit\'e + bbox dessin\'ees\\
   \scriptsize Onglet 2 : IoU du mod\`ele\\
   \scriptsize \texttt{st.fragment} non bloquant};

% couches sous le consumer
\node[store, below=1.5cm of cons.south west, anchor=north, xshift=8mm] (staging)
  {\textbf{STAGING/}\\Parquet\\\scriptsize historique images\\\scriptsize propres};
\node[store, below=1.5cm of cons.south east, anchor=north, xshift=-8mm] (ckpt)
  {\textbf{CHECKPOINT/}\\\scriptsize exactly-once};
\node[store, below=1.5cm of dest1] (arch)
  {\textbf{ARCHIVE/}\\\scriptsize DEST purg\'e apr\`es\\\scriptsize persistance};

% scheduler
\node[note, above=0.7cm of prod1]
  (sched) {\textbf{Trigger externe} (\texttt{main.py --producer-every 60s},\\cron/schtasks) : relance le job \'eph\'em\`ere};

% note idempotence
\node[note, above=0.7cm of out] (idem)
  {\textbf{Sink idempotent}\\\texttt{ingest\_date=YYYY-MM-DD/}\\\texttt{batch\_id=N/\_SUCCESS}\\rejeu $\Rightarrow$ skip, 0 doublon};

% ════════════════ MOD\`ELES (milieu) ════════════════
\node[store, below=3.4cm of out, fill=red!8, draw=red!60!black] (models)
  {\textbf{MODELS/}\\\texttt{plate\_locator.onnx}\\\scriptsize YOLOv8n $\cdot$ opset 17\\\scriptsize $[1,3,256,256]\!\to\![1,5,N]$};
\node[store, right=1.2cm of models, fill=red!8, draw=red!60!black] (metrics)
  {\textbf{metrics/}\\\texttt{<version>.json}\\\scriptsize IoU moy/m\'ed $\cdot$ \%$\geq$0.5};

% ════════════════ FLUX HEBDOMADAIRE (bas) ════════════════
\node[store, below=7.2cm of src1] (src2)
  {\textbf{SOURCE/}\\\texttt{ml\_dataset/}\\\scriptsize\texttt{\{train,test,val\}/}\\\scriptsize\texttt{\{images,labels\}/}};

\node[mlserv, right=1.5cm of src2] (prod2)
  {\textbf{PRODUCER}\\\scriptsize mode \texttt{batch} (hebdo)\\\scriptsize copie r\'ecursive int\'egrale\\\scriptsize idempotent $\cdot$ puis exit 0};

\node[store, right=1.5cm of prod2] (dest2)
  {\textbf{DEST/}\\\texttt{ml\_dataset/}\\\scriptsize copie fig\'ee, isol\'ee\\\scriptsize du stream};

\node[mlserv, right=1.7cm of dest2, minimum width=4.2cm] (yolo)
  {\textbf{ML TRAINING}\\\texttt{ml/train\_yolo.py} --- Python\\[2pt]
   \scriptsize fine-tune \textbf{YOLOv8n} (mono-classe)\\
   \scriptsize \'eval IoU vs ground truth\\
   \scriptsize d\'eclench\'e 1$\times$/semaine};

% ════════════════ FL\`ECHES flux continu ════════════════
\draw[cflow] (src1) -- node[step, above=1mm, pos=0.5, yshift=2mm]{1}
             node[lbl, below]{READ} (prod1);
\draw[cflow] (prod1) -- node[step, pos=0.5, yshift=4mm]{2}
             node[lbl, below]{WRITE\\\scriptsize(d\'eplacement)} (dest1);
\draw[cflow] (dest1) -- node[step, pos=0.5, yshift=4mm]{3}
             node[lbl, below]{FETCH\\\scriptsize stream} (cons);
\draw[cflow] (cons.south -| staging.north) -- node[step, pos=0.4, xshift=-4mm]{4}
             node[lbl, right=1mm, pos=0.45]{WRITE\\\scriptsize AVANT predict} (staging);
\draw[cflow] (cons) -- node[step, pos=0.45, yshift=4mm]{7}
             node[lbl, below]{PREDICT\\\scriptsize+ WRITE} (out);
\draw[cflow] (out) -- node[step, pos=0.5, yshift=4mm]{8}
             node[lbl, below]{READ\\\scriptsize TTL 30 s} (dash);
\draw[cflow] (cons.south -| ckpt.north) -- (ckpt);
\draw[cflow] (dest1) -- node[lbl, left]{ARCHIVE\\\scriptsize apr\`es persistance} (arch);
\draw[gray!60, dashed, thick, -{Stealth}] (sched) -- (prod1);

% ════════════════ FL\`ECHES flux hebdo ════════════════
\draw[wflow] (src2) -- node[step, pos=0.5, yshift=4mm]{1}
             node[lbl, below]{READ\\\scriptsize r\'ecursif} (prod2);
\draw[wflow] (prod2) -- node[step, pos=0.5, yshift=4mm]{2}
             node[lbl, below]{WRITE} (dest2);
\draw[wflow] (dest2) -- node[step, pos=0.5, yshift=4mm]{5}
             node[lbl, below]{READ\\\scriptsize images + labels} (yolo);
\draw[wflow] (yolo.east) -- ++(0.8,0) |- node[step, pos=0.75, yshift=4mm]{6}
             node[lbl, pos=0.75, below=2mm]{EXPORT ONNX} (models.west);
\draw[wflow] (yolo.north east) ++(-0.4,0) |- (metrics.east) ;

% ════════════════ FL\`ECHES mod\`ele ════════════════
\draw[mflow] (models) -- node[step, pos=0.55, xshift=5mm]{7}
             node[lbl, left, pos=0.5]{broadcast au\\d\'emarrage\\\scriptsize 1 session ONNX\\\scriptsize par partition} (cons.south);
\draw[mflow] (metrics.east) -- ++(0.9,0) |- node[lbl, pos=0.8, above]{READ 8 --- onglet 2} (dash.south east);

% ════════════════ ZONES de fond ════════════════
\begin{scope}[on background layer]
  \node[fit=(src1)(prod1)(dest1)(cons)(out)(dash)(sched)(idem)(staging)(ckpt)(arch),
        fill=blue!4, rounded corners=6pt, inner sep=10pt,
        label={[blue!50!black, font=\bfseries]north west:FLUX CONTINU --- inf\'erence au fil de l'eau}] {};
  \node[fit=(src2)(prod2)(dest2)(yolo),
        fill=orange!6, rounded corners=6pt, inner sep=10pt,
        label={[orange!80!black, font=\bfseries]north west:FLUX HEBDOMADAIRE --- r\'eentra\^inement planifi\'e}] {};
  \node[fit=(models)(metrics),
        fill=red!4, rounded corners=6pt, inner sep=8pt,
        label={[red!60!black, font=\bfseries\scriptsize]north west:REGISTRE DE MOD\`ELES}] {};
\end{scope}

% ════════════════ L\'egende ════════════════
\matrix[draw=gray!50, rounded corners=2pt, fill=white, inner sep=5pt,
        below=1.2cm of src2.south west, anchor=north west,
        column sep=3mm, row sep=1.5mm, font=\scriptsize] {
  \draw[cflow] (0,0) -- (0.7,0); & \node[align=left]{flux continu (permanent c\^ot\'e Consumer)}; \\
  \draw[wflow] (0,0) -- (0.7,0); & \node[align=left]{flux hebdomadaire (jobs \'eph\'em\`eres)}; \\
  \draw[mflow] (0,0) -- (0.7,0); & \node[align=left]{mod\`ele ONNX / m\'etriques}; \\
  \node[step]{n}; & \node[align=left]{\'etapes (1)--(8) du sch\'ema d'architecture}; \\
};

\end{tikzpicture}
\end{document}
```
