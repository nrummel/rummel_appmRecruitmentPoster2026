%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Originally Commissioned by the
%% AMS Graduate Student Chapter at CMU
%%
%% Then modified for academic posters in LaTeX at
%% Michigan State University
%%
%% Created by Rachel Domagalski and Dr. Patrick Davis
%%
%% Redesigned for use at 
%% Again edited by Nic Rummel for use at CU Boulder
%%
%% Original Version: 21 September 2016
%% Last Updated:  22 Feb 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rummelPoster}[2025/02/22]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Process Class Options            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{xparse}
\RequirePackage{etoolbox}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions \relax
\LoadClass[t]{beamer}

\newcommand{\myskip}{5pt}
\newcommand{\mybigskip}{5pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%          Preamble Definitions              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NewDocumentCommand\@advisor{}{} % SET DEFAULT TYPE
\NewDocumentCommand\advisor{O{1} m}
    {\RenewDocumentCommand\@advisor{}{\par\vspace{\mybigskip}\Large Advisor\ifnumgreater{#1}{1}{s}{}: #2}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%          Import Standard Packages          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage{amsmath,amssymb,amsfonts,amstext,amscd,mathtools,exscale}
\RequirePackage{xcolor}
\RequirePackage{graphicx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%        Define the Official  Colors      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% From Brand Identity Standards and Web Style Guide 
%% PRIMARY COLORS (backgrounds, headlines, buttons, containers)
\definecolor{green}{HTML}{18453B}
\definecolor{white}{HTML}{FFFFFF}

%% SECONDARY COLORS (backgounds for pages, containers, buttons, etc)
\definecolor{brightgreen}{HTML}{0DB14B}
\definecolor{grey}{HTML}{97A2A2}
\definecolor{brightorange}{HTML}{F08521}
\definecolor{teal}{HTML}{008183}
\definecolor{blue}{HTML}{909AB7}
\definecolor{black}{HTML}{535054}
\definecolor{yellow}{HTML}{D1DE3F}
\definecolor{biege}{HTML}{E8D9B5}
\definecolor{lightorange}{HTML}{C89A58}
\definecolor{olive}{HTML}{94AE4A}
\definecolor{cugold}{HTML}{CFB87C}
\definecolor{red}{HTML}{CB5A28}

%  TEXT COLORbeamercolortheme
\definecolor{text}{HTML}{333333}

%% Self Defined (not official, different saturation of official colors) %%
\definecolor{lightgreen}{HTML}{ecf8f6}
\definecolor{darkgreen}{HTML}{0D2620}
\definecolor{brightlightgreen}{HTML}{e7fdf0}
\definecolor{lighterpurple}{HTML}{ffe6fc}
\definecolor{lightgrey}{HTML}{e2e2e2}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Set Global Properties            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Page Layout
\RequirePackage[orientation=landscape,size=custom,width=40,height=32,scale=.5]{beamerposter}
\setbeamercolor{background canvas}{bg=lightgrey}
\setbeamertemplate{navigation symbols}{}
%% Change Default List Styles
\setbeamertemplate{itemize items}[square]
\setbeamertemplate{enumerate items}[default]
\setbeamercolor{item}{fg=black,bg=white}

%% Change Default Caption Styles
\setbeamertemplate{caption}[numbered]{}
\setbeamercolor{caption name}{fg=black}

%% Change Default Bibliography Styles
\setbeamertemplate{bibliography item}[text]
\setbeamercolor{bibliography item}{fg=black,bg=black}
\setbeamercolor*{bibliography entry title}{fg=black,bg=black}
\setbeamercolor*{bibliography entry author}{fg=black,bg=black}
\setbeamercolor*{bibliography entry journal}{fg=black,bg=black}
\setbeamercolor*{bibliography entry note}{fg=black,bg=black}
\setbeamercolor*{bibliography entry location}{fg=black,bg=black}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Reformat the Blocks              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% block
%\setbeamercolor{block title}{fg=white,bg=green}
\setbeamercolor{block title}{fg=white,bg=black}
\setbeamercolor{block body}{fg=black,bg=white}
\AfterEndEnvironment{block}{\vspace{\myskip}}

\setbeamertemplate{block begin}{
  \vskip.75ex
  \begin{beamercolorbox}[leftskip=0cm,colsep*=.75ex]{block title}%
  \usebeamerfont*{block title}\insertblocktitle
  \end{beamercolorbox}%
  {\ifbeamercolorempty[bg]{block body}{}{\nointerlineskip\vskip-0.5pt}}%
  \usebeamerfont{block body}%
  \begin{beamercolorbox}[colsep*=.75ex,sep=.5ex,vmode]{block body}%
  \ifbeamercolorempty[bg]{block body}{\vskip-.25ex}{\vskip-.75ex}\vbox{}%
}
\setbeamertemplate{block end}{\end{beamercolorbox}}

%% alertedblock
\setbeamercolor{block title alerted}{fg=white,bg=brightgreen}
\setbeamercolor{block body alerted}{fg=black,bg=white}

\AfterEndEnvironment{alertblock}{\vspace{20pt}}

\setbeamertemplate{block alerted begin}{
  \vskip.75ex
  \begin{beamercolorbox}[leftskip=1cm,colsep*=.75ex]{block title alerted}%
  \usebeamerfont*{block title}\insertblocktitle
  \end{beamercolorbox}%
  {\ifbeamercolorempty[bg]{block body}{}{\nointerlineskip\vskip-0.5pt}}%
  \usebeamerfont{block body}%
  \begin{beamercolorbox}[colsep*=.75ex,sep=.75ex,vmode]{block body}%
  \ifbeamercolorempty[bg]{block body}{\vskip-.25ex}{\vskip-.75ex}\vbox{}%
}
\setbeamertemplate{block alerted end}{\end{beamercolorbox}}

%% exampleblock
\setbeamercolor{block title example}{fg=black,bg=cugold}
\setbeamercolor{block body example}{fg=black,bg=white}

\AfterEndEnvironment{exampleblock}{\vspace{20pt}}

\setbeamertemplate{block example begin}{
  \vskip.75ex
  \begin{beamercolorbox}[leftskip=1cm,colsep*=.75ex]{block title example}%
  \usebeamerfont*{block title}\insertblocktitle
  \end{beamercolorbox}%
  {\ifbeamercolorempty[bg]{block body}{}{\nointerlineskip\vskip-0.5pt}}%
  \usebeamerfont{block body}%
  \begin{beamercolorbox}[colsep*=.75ex,sep=.75ex,vmode]{block body}%
  \ifbeamercolorempty[bg]{block example body}{\vskip-.25ex}{\vskip-.75ex}\vbox{}%
}
\setbeamertemplate{block example end}{\end{beamercolorbox}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Define the Headline              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setbeamercolor{headline}{fg=black,bg=white}

\setbeamertemplate{headline}{  
\leavevmode
\begin{beamercolorbox}[wd=\paperwidth]{headline}
  \vspace*{\mybigskip}
  \begin{columns}[c]
  \begin{column}{.7\paperwidth}
    \begin{flushleft}
      {\veryHuge\bfseries\inserttitle}\par\vspace{\mybigskip}
      {\LARGE\insertauthor}\par\vspace{\myskip}
      {\Large\insertinstitute}
      {\@advisor}
    \end{flushleft}
  \end{column}
  \begin{column}{.22\paperwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/CU_Boulder_logo.png}
    \textbf{References}
    % \scriptsize
    \bibliographystyle{unsrtnat}
    \bibliography{recruitmentPoster2026.bib}
    % \begin{end}
  \end{column}
  \end{columns}
  \vspace*{\mybigskip}
\end{beamercolorbox}
}

