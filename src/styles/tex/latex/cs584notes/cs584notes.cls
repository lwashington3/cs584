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

\LoadClass[12pt,twocolumn]{report}

\RequirePackage[title={\ifnum\csnotes@number=0\csnotes@number\ - \fi\csnotes@title}]{cs584}
\RequirePackage{enumitem}
\RequirePackage[most]{tcolorbox}
\RequirePackage{mdsymbol}

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

\let\eq\equation
\let\endeq\endequation
\def\eqs{\equation*}
\def\endeqs{\endequation*}

\let\al\aligned
\let\endal\endaligned

\let\gat\gather
\let\endgat\endgather

\definecolor{eqred}{HTML}{ff0000}
\definecolor{eqpurple}{HTML}{7030a0}
\definecolor{eqblue}{HTML}{0000ff}

\renewenvironment{equation}[1][eqred]
{%
	\color{#1}%
	\eq%
}{%
	\endeq%
}

\renewenvironment{aligned}[1][eqred]
{%
	\color{#1}%
	\al%
}{%
	\endal%
}

\renewenvironment{gather}[1][eqred]
{%
	\color{#1}%
	\gat%
}{%
	\endgat%
}

%\renewenvironment{equation*}[1][eqred]
%{%
%	\color{#1}%
%	\eqs%
%}{%
%	\endeqs%
%}

\definecolor{lightgreen}{HTML}{e9f1e8}
\definecolor{medgreen}{HTML}{d0e1cd}
\definecolor{darkgreen}{HTML}{4ea72e}

\definecolor{lightblue}{HTML}{00b0f0}

\newcommand{\tableheader}[1]{\textbf{\textcolor{white}{#1}}}

\newcommand{\tildehat}[1]{\hat{#1}} % TODO: Make the tildehat actuaDlly have a tilde
