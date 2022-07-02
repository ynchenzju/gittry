filetype off
call pathogen#infect()
filetype plugin on
set tabstop=4
set shiftwidth=4
set ai
set si
set nu
set nobackup
syntax on
filetype on
set nocompatible
set nocscopeverbose
set ruler
set expandtab
set hlsearch
set cursorline
set backspace=indent,eol,start
set t_Co=256
"colorscheme molokai
syntax enable
set background=dark
colorscheme solarized
set nocp
set fileencodings=utf-8,gbk,gb18030
" set encoding=gbk
set encoding=utf-8
filetype plugin indent on
"hi CursorLine term=underline cterm=underline
language en_US

let anyfold_activate=1
set foldlevel=5

let g:Powerline_symbols = 'unicode'
set fillchars+=stl:\ ,stlnc:\
set laststatus=2
let g:airline_powerline_fonts = 1
let g:Powerline_dividers_override = 2
let g:Powerline_dividers_override = ['>>', '>', '<<', '<']
set linespace=0
set scrolloff=2

"取消自动备份和swp文件
set nobackup
set nowb
set noswapfile
"当文件被外部改变时自动读取
set autoread
"set paste

set nocompatible
set cursorline
let g:pymode_trim_whitespaces = 1
let g:pymode_options_colorcolumn = 1
let g:pymode_indent = 1
let g:pymode_lint_on_write = 1
let g:pymode_lint_on_fly = 0
let g:pymode_lint_message = 1
let g:pymode_options_max_line_length =100
let g:pymode_folding = 0
highlight ColorColumn ctermbg=6
set colorcolumn=100
highlight ExtraWhitespace ctermbg=red guibg=red

autocmd BufRead *.cpp :call bdswitch#highlight()
autocmd BufRead *.h :call bdswitch#highlight()
autocmd filetype cpp :call bdswitch#highlight()

set mouse-=a
set tags=./.tags;,.tags
set cscopequickfix=s-,c-,d-,i-,t-,e-
"vim-commentary
"为python和shell等添加注释
autocmd FileType python,shell,coffee set commentstring=#\ %s
"修改注释风格
autocmd FileType java,c,cpp set commentstring=//\ %s

let mapleader="r"
"imap <C-j> <Esc><C-e>a
"imap <C-s> <Esc>:w!<CR>i
map <C-j> <C-e>
noremap <C-k> <C-y>
"nmap <silent> <C-y> <Plug>(ac-smooth-scroll-c-d)
noremap <C-y> <C-d>
map <C-h> gt
map <C-l> gT
map = <C-w>>
map - <C-w><
nmap <Space> <C-w>
nmap <BS> X
map <S-l> <Esc>v$ho
map <S-h> <Esc>v^o
" imap <S-l> <Esc>lv$ho
" imap <S-h> <Esc>lv^o
nmap df daw
nmap d6 d^
nmap d4 d$
nmap y6 y^
nmap y4 y$
nmap e el
map f <Plug>Sneak_;
map q <Plug>Sneak_,
map Q :q<CR>
noremap 1 3<C-e>
noremap 2 3<C-y>
"nmap <C-f> :Grep<Space>
"tagbar
map tb :TagbarToggle<CR>
let g:tagbar_left=1
let g:tagbar_width=30
"Leaderf
map <leader>h :LeaderfLine
map <leader>l :LeaderfFunction
let g:Lf_ShortcutF = '<C-P>'
"tmux
if &term =~ '256color'
    set t_ut=
endif

"gutentags
let g:gutentags_project_root = ['.root', '.svn', '.git', '.hg', '.project']
" 所生成的数据文件的名称
let g:gutentags_ctags_tagfile = '.tags'
" 将自动生成的 tags 文件全部放入 ~/.cache/tags 目录中，避免污染工程目录
let s:vim_tags = expand('~/.cache/tags')
let g:gutentags_cache_dir = s:vim_tags
" 配置 ctags 的参数
let g:gutentags_ctags_extra_args = ['--fields=+niazS', '--extra=+q']
let g:gutentags_ctags_extra_args += ['--c++-kinds=+px']
let g:gutentags_ctags_extra_args += ['--c-kinds=+px']
let g:gutentags_ctags_extra_args += ['--exclude=output*']
" 检测 ~/.cache/tags 不存在就新建
if !isdirectory(s:vim_tags)
    silent! call mkdir(s:vim_tags, 'p')
endif


"vimdiff
set laststatus=2 "show the status line
set statusline=%-10.3n  "buffer number
map <silent> <leader>1 :diffget 1<CR> :diffupdate<CR>
map <silent> <leader>2 :diffget 2<CR> :diffupdate<CR>
map <silent> <leader>3 :diffget 3<CR> :diffupdate<CR>
map <silent> <leader>4 :diffget 4<CR> :diffupdate<CR>

"neocomplete
"let g:neocomplete#enable_auto_select = 1
"" Disable AutoComplPop.
"let g:acp_enableAtStartup = 0
"" Use neocomplete.
"let g:neocomplete#enable_at_startup = 1
"" Use smartcase.
"let g:neocomplete#enable_smart_case = 1
"" Set minimum syntax keyword length.
"let g:neocomplete#sources#syntax#min_keyword_length = 3

"" Define dictionary.
"let g:neocomplete#sources#dictionary#dictionaries = {
"    \ 'default' : '',
"    \ 'vimshell' : $HOME.'/.vimshell_hist',
"    \ 'scheme' : $HOME.'/.gosh_completions'
"        \ }

"" Define keyword.
"if !exists('g:neocomplete#keyword_patterns')
"    let g:neocomplete#keyword_patterns = {}
"endif
"let g:neocomplete#keyword_patterns['default'] = '\h\w*'

"" Plugin key-mappings.
"inoremap <expr><C-g>     neocomplete#undo_completion()
"inoremap <expr><C-l>     neocomplete#complete_common_string()

"" Recommended key-mappings.
"" <CR>: close popup and save indent.
"inoremap <silent> <CR> <C-r>=<SID>my_cr_function()<CR>
"function! s:my_cr_function()
"  return (pumvisible() ? "\<C-y>" : "" ) . "\<CR>"
"  " For no inserting <CR> key.
"  "return pumvisible() ? "\<C-y>" : "\<CR>"
"endfunction
"" <TAB>: completion.
"inoremap <expr><TAB>  pumvisible() ? "\<C-n>" : "\<TAB>"
"" <C-h>, <BS>: close popup and delete backword char.
"inoremap <expr><C-h> neocomplete#smart_close_popup()."\<C-h>"
"inoremap <expr><BS> neocomplete#smart_close_popup()."\<C-h>"
"" Close popup by <Space>.
""inoremap <expr><Space> pumvisible() ? "\<C-y>" : "\<Space>"
"inoremap <expr><CR> pumvisible()? "\<C-y>" : "\<CR>"

"" AutoComplPop like behavior.
""let g:neocomplete#enable_auto_select = 1

"" Shell like behavior(not recommended).
""set completeopt+=longest
""let g:neocomplete#enable_auto_select = 1
""let g:neocomplete#disable_auto_complete = 1
""inoremap <expr><TAB>  pumvisible() ? "\<Down>" : "\<C-x>\<C-u>"

"" Enable omni completion.
"autocmd FileType css setlocal omnifunc=csscomplete#CompleteCSS
"autocmd FileType html,markdown setlocal omnifunc=htmlcomplete#CompleteTags
"autocmd FileType javascript setlocal omnifunc=javascriptcomplete#CompleteJS
"autocmd FileType python setlocal omnifunc=pythoncomplete#Complete
"autocmd FileType xml setlocal omnifunc=xmlcomplete#CompleteTags

"" Enable heavy omni completion.
"if !exists('g:neocomplete#sources#omni#input_patterns')
"  let g:neocomplete#sources#omni#input_patterns = {}
"endif
""let g:neocomplete#sources#omni#input_patterns.php = '[^. \t]->\h\w*\|\h\w*::'
""let g:neocomplete#sources#omni#input_patterns.c = '[^.[:digit:] *\t]\%(\.\|->\)'
""let g:neocomplete#sources#omni#input_patterns.cpp = '[^.[:digit:] *\t]\%(\.\|->\)\|\h\w*::'

"" For perlomni.vim setting.
"" https://github.com/c9s/perlomni.vim
"let g:neocomplete#sources#omni#input_patterns.perl = '\h\w*->\h\w*\|\h\w*::'

" 这句话会导致按l延迟
" noremap <silent> lfcw :LeaderfFunctionCword<CR>

" OmniCppComplete
let OmniCpp_NamespaceSearch = 1
let OmniCpp_GlobalScopeSearch = 1
let OmniCpp_ShowAccess = 1
let OmniCpp_ShowPrototypeInAbbr = 1 " 显示函数参数列表
let OmniCpp_MayCompleteDot = 1   " 输入 .  后自动补全
let OmniCpp_MayCompleteArrow = 1 " 输入 -> 后自动补全
let OmniCpp_MayCompleteScope = 1 " 输入 :: 后自动补全
let OmniCpp_DefaultNamespaces = ["std", "_GLIBCXX_STD"]
let OmniCpp_SelectFirstItem = 2
" 自动关闭补全窗口
au CursorMovedI,InsertLeave * if pumvisible() == 0|silent! pclose|endif
set completeopt=menuone,menu,longest"
highlight Pmenu    guibg=darkgrey  guifg=black
highlight PmenuSel guibg=lightgrey guifg=black

" set statusline+=%#warningmsg#
" set statusline+=%{SyntasticStatuslineFlag()}
" set statusline+=%*
" let g:syntastic_always_populate_loc_list = 1
" let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

call pymode#default('g:pymode_rope_autoimport', 0)
let g:pymode_rope = 0

"使用语法高亮定义代码折叠
set foldmethod=syntax
"打开文件是默认不折叠代码
set foldlevelstart=99
