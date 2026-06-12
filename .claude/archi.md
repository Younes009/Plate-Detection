\documentclass[tikz,border=10mm]{standalone}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}
\usepackage{amsmath}
\usepackage{xcolor}

% Définition de la palette de couleurs basée sur le schéma
\definecolor{sparkgreen}{HTML}{7FFF00}       % Vert vif (Spark Core / Structured Streaming)
\definecolor{pythonorange}{HTML}{DEB887}    % Marron clair / Sable (Python)
\definecolor{predictpurple}{HTML}{DA70D6}    % Violet / Orchidée (Predict)
\definecolor{streamlitblue}{HTML}{87CEEB}    % Bleu ciel (Streamlit)

\begin{document}
\begin{tikzpicture}[
node distance=2cm and 2.5cm,
>=Stealth,
font=\sffamily\small,
% Définition des styles d'éléments
database/.style={
cylinder, cylinder uses custom aspect, cylinder to end engine,
draw, shape border rotate=90, minimum width=0.8cm, minimum height=1cm,
line width=0.8pt
},
queue/.style={
draw, minimum width=1.2cm, minimum height=0.8cm,
rectangle, line width=0.8pt,
append after command={
\pgfextra{
\draw (\pathlastnode.north west) -- (\pathlastnode.north east);
\draw (\pathlastnode.south west) -- (\pathlastnode.south east);
\draw ([yshift=2.5mm]\pathlastnode.south west) -- ([yshift=2.5mm]\pathlastnode.south east);
\draw ([yshift=-2.5mm]\pathlastnode.north west) -- ([yshift=-2.5mm]\pathlastnode.north east);
}
}
},
service/.style={
circle, draw, minimum size=2.2cm, align=center, line width=1pt, fill=white
},
decision/.style={
diamond, draw, minimum width=1.5cm, minimum height=1.5cm, align=center, line width=0.8pt
},
badge/.style={
draw, fill=#1, rounded corners=3pt, font=\sffamily\bfseries\scriptsize, inner sep=3pt, align=center
},
labelstyle/.style={
circle, draw, inner sep=1pt, font=\sffamily\bfseries\scriptsize, fill=white
}
]

    % --- NIVEAU 1 : SOURCE & MICROSERVICE 1 ---
    \node (source_label) at (0, 3.5) {\textbf{... /SOURCE/}};
    \node[database] (source) at (0, 2.5) {};
    
    \node[badge=sparkgreen] (spark1) at (2, 2.8) {SPARK CORE\\SCALA};
    \node[service] (ms1) at (2, 1) {\textbf{MICRO}\\\textbf{SERVICE}\\\textbf{1}};
    
    % Attributs de Microservice 1 (Producer)
    \node[anchor=west, align=left, font=\sffamily\scriptsize] (ms1_attr) at (1.6, -0.7) {
        \textbf{PRODUCER}\\
        $\vdash$ BATCH SIZE\\
        $\vdash$ INTERVAL\\
        $\vdash$ RECURSIVE
    };

    % --- NIVEAU 2 : DESTINATION (QUEUE) ---
    \node (dest_label) at (5, 3.5) {\textbf{... /DEST/}};
    \node[queue] (dest) at (5, 2.5) {};

    % --- NIVEAU 3 : MICROSERVICE 2 ---
    \node[badge=sparkgreen] (spark2) at (6.5, 1.8) {STRUCTURED\\STREAMING\\SCALA};
    \node[service] (ms2) at (6.5, -0.3) {\textbf{MICRO}\\\textbf{SERVICE}\\\textbf{2}};
    
    % Attributs de Microservice 2 (Consumer)
    \node[anchor=west, align=left, font=\sffamily\scriptsize] (ms2_attr) at (6.1, -2.3) {
        \textbf{CONSUMER}\\
        $\vdash$ READ\\
        $\vdash$ PARSE\\
        $\vdash$ TRANSFORM\\
        $\vdash$ SCORE\\
        $\vdash$ \colorbox{predictpurple!40}{\textbf{PREDICT}}\\
        $\vdash$ WRITE
    };

    % --- NIVEAU 4 : MACHINE LEARNING SERVICE & MODEL ---
    \node[badge=pythonorange] (python) at (6.5, -4.7) {PYTHON};
    \node[service] (ml_service) at (6.5, -6) {\textbf{MACHINE}\\\textbf{LEARNING}\\\textbf{SERVICE}};
    
    % Attributs du service ML
    \node[anchor=west, align=left, font=\sffamily\scriptsize] (ml_attr) at (6.2, -7.5) {
        \textbf{$\vdash$ TRAIN}\\
        \textbf{$\vdash$ TEST}\\
        \textbf{$\vdash$ VAL}
    };

    \node[decision] (model) at (9.5, -4.5) {\textbf{MODEL}};

    % --- NIVEAU 5 : OUTPUT & STREAMLIT ---
    \node (output_label) at (2.5, -3.5) {\textbf{... /OUTPUT/}};
    \node[database] (output) at (2.5, -4.5) {};

    \node[badge=streamlitblue] (streamlit_badge) at (0.5, -5.5) {STREAMLIT};
    \node[service] (vis) at (0.5, -6.8) {\textbf{DATA}\\\textbf{VISUALI-}\\\textbf{ZATION}};


    % --- ARCS, FLÈCHES ET NUMÉROTATIONS (ÉTAPES) ---
    
    % 1. Read: Source -> MS1
    \draw[->, line width=0.8pt] (source.south) |- (ms1.west) 
        node[pos=0.7, labelstyle, label=below:{\scriptsize READ}] {1};

    % 2. Write: MS1 -> Dest
    \draw[->, line width=0.8pt] (ms1.east) -| (dest.south)
        node[pos=0.3, labelstyle, label=below:{\scriptsize WRITE}] {2};

    % 3. Fetch: Dest -> MS2
    \draw[->, line width=0.8pt] (dest.east) -- ++(1.8,0) |- (ms2.east)
        node[pos=0.25, labelstyle, label=right:{\scriptsize FETCH}] {3};

    % 4. Write: MS2 -> Output
    \draw[->, line width=0.8pt] (ms2.west) |- (output.north)
        node[pos=0.3, labelstyle, label=left:{\scriptsize WRITE}] {4};

    % 5. Read: Output -> ML Service
    \draw[->, line width=0.8pt] (output.south) |- (ml_service.west)
        node[pos=0.7, labelstyle, label=left:{\scriptsize READ}] {5};

    % 6. Export Model: ML Service -> Model Block
    \draw[->, line width=0.8pt] (ml_service.east) -| (model.south)
        node[pos=0.3, labelstyle, label=right:{\scriptsize\begin{tabular}{l}EXPORT\\$\vdash$ ONNX\\$\vdash$ PICKLE\end{tabular}}] {6};

    % 7. Read Model: Model Block -> MS2
    \draw[->, line width=0.8pt] (model.north) |- (ms2.355)
        node[pos=0.7, labelstyle, label=right:{\scriptsize READ}] {7};

    % 8. Read: Output -> Streamlit Visualization
    \draw[->, line width=0.8pt] (output.west) |- (vis.east)
        node[pos=0.7, labelstyle, label=below:{\scriptsize READ}] {8};

\end{tikzpicture}
\end{document}