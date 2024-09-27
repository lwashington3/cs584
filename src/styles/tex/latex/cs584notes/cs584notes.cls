%! Class = cs584notes
%! Author = Len Washington III
%! Date = 8/21/24

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cs584notes}[Len Washington III's Document Class]

\RequirePackage{kvoptions}

\SetupKeyvalOptions{
	family=csnotes,
	prefix=csnotes@
}

\DeclareStringOption{title}
\DeclareStringOption{number}

\ProcessKeyvalOptions*

\LoadClass[12pt]{report}

\RequirePackage[title={\csnotes@number\ - \csnotes@title}]{cs584}
\RequirePackage{enumitem}
\RequirePackage[most]{tcolorbox}

\setcounter{chapter}{\csnotes@number}
\addtocounter{chapter}{-1}

\definecolor{emphblue}{HTML}{0629ff}

\renewcommand{\emph}[1]{\textcolor{emphblue}{#1}}
\newcommand{\definition}[2]{\textcolor{emphblue}{#1}: #2}
\newcommand{\data}[1]{\textcolor{red}{#1}}

\newcommand{\sign}{\mbox{sign}}
\let\oldvec\vec
\renewcommand{\vec}[1]{%
	\mathbf{#1}%
}

\setlength{\parindent}{0in}

\definecolor{svmgreen}{HTML}{008000}
\newenvironment{svmbox}
{
	\begin{tcolorbox}[colback=white,colframe=svmgreen]
}{
	\end{tcolorbox}
}

\renewcommand{\maketitle}{%
%	{\Huge\textbf{\cs@title}}
	\chapter{\csnotes@title}\label{ch:\csnotes@number}
}

\let\begindoc\document
\let\enddoc\enddocument
\renewenvironment{document}{
	\begindoc
	\maketitle
	\tableofcontents
}{
	\enddoc
}