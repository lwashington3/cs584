%! Class = cs584exam
%! Author = Len Washington III
%! Date = 10/4/24

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cs584exam}[Len Washington III's Document Class]

\LoadClass[twocolumn,number={0},title={{CS 584 Fall 2024 Midterm Cheat Sheet}}]{cs584notes}

\RequirePackage{kvoptions}
\SetupKeyvalOptions{
	family=csexam,
	prefix=csexam@
}

\DeclareStringOption{exam}
\DeclareStringOption[6pt]{fontsize}
\DeclareBoolOption[true]{color}
\DeclareBoolOption[true]{compact}
\DeclareBoolOption[true]{showchapter}
\DeclareStringOption[0.3]{linespread}

\ProcessKeyvalOptions*

\RequirePackage{enumitem}
\RequirePackage{bookmark}
\RequirePackage{alphalph}
\RequirePackage{multirow}
\RequirePackage[fontsize=\csexam@fontsize]{fontsize}

\setlength{\parindent}{0pt}

\let\ifcolor\ifcsexam@color

\ifcolor%
\renewcommand{\emph}[2][red]{\textcolor{#1}{#2}}%
\else%
\renewcommand{\emph}[2][]{\textbf{#2}}%
\fi

\renewcommand{\definition}[3][red]{\textbf{\textcolor{\ifcolor%
#1\else%
black\fi}{#2}} -- #3}

\newlength{\verticalmargin}
\setlength{\verticalmargin}{1em}
\newlength{\horizontalmargin}
\setlength{\horizontalmargin}{1em}
\setlength{\parskip}{0cm}

\geometry{top=\verticalmargin, bottom=\verticalmargin, left=\horizontalmargin, right=\horizontalmargin}

\renewenvironment{document}{
	\begindoc
}{
	\enddoc
}

\setlength{\itemsep}{-0.5em}

\RequirePackage{enumitem}
\newlist{problems}{enumerate}{2}
\setlist[problems, 1]{
	font=\textbf,
	label=\Alph{problemsi}),
	leftmargin=30pt,
}

\setlist[problems, 2]{
	label=\textbf{(\alph{problemsii})},
	leftmargin=30pt,
}

\linespread{\csexam@linespread}

\newcommand{\defineexamcolor}[2]{%
	\ifcolor
	\definecolor{#1}{HTML}{#2}
	\else
	\definecolor{#1}{HTML}{000000}
	\fi
}

\ifcsexam@compact%
\RequirePackage[compact]{titlesec}
\else%
\RequirePackage{titlesec}
\fi%

\titleformat{\chapter}[block]
{\large\bfseries}
{\ifcsexam@showchapter{}Chapter \arabic{chapter}: \fi}
{0em}
{}

\titleformat{\section}[runin]
{\bfseries}
{\ifcsexam@showchapter{}(\arabic{chapter}.\arabic{section}): \fi}
{0em}
{}

\titleformat{\subsection}[runin]
{\bfseries}
{\ifcsexam@showchapter{}(\arabic{chapter}.\arabic{section}.\arabic{subsection}): \fi}
{0em}
{}

\renewcommand{\chapter}[1]{\stepcounter{chapter}}
