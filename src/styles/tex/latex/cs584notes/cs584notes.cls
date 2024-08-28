%! Class = cs584notes
%! Author = Len Washington III
%! Date = 8/21/24

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cs584notes}[Len Washington III's Document Class]

\LoadClass[12pt]{report}

\RequirePackage{cs584}
\RequirePackage{enumitem}

\definecolor{emphblue}{HTML}{0629ff}

\renewcommand{\emph}[1]{\textcolor{emphblue}{#1}}
\newcommand{\definition}[2]{\textcolor{emphblue}{#1}: #2}
\newcommand{\data}[1]{\textcolor{red}{#1}}

\newcommand{\sign}{\mbox{sign}}
\let\oldvec\vec
\renewcommand{\vec}[1]{%
	\mathbf{#1}%
}

\newenvironment{svmbox}
{
	\begin{center}
}{
	\end{center}
}