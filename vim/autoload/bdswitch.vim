
function! bdswitch#update()
    !if [ \! -d $HOME'/.vim/autoload/.asconf' ]; then mkdir -p ~/.vim/autoload/.asconf/mixer; mkdir -p ~/.vim/autoload/.asconf/ranker; svn co https://svn.baidu.com/app/ecom/im/trunk/production/prod/ranker/conf ~/.vim/autoload/.asconf/ranker; svn co https://svn.baidu.com/app/ecom/im/trunk/production/prod/mixer/conf ~/.vim/autoload/.asconf/mixer; svn up ~/.vim/autoload/.asconf/mixer;svn up ~/.vim/autoload/.asconf/ranker; else svn up ~/.vim/autoload/.asconf/mixer;svn up ~/.vim/autoload/.asconf/ranker; fi

    call bdswitch#init()
    call bdswitch#highlight()
endfunction


function! bdswitch#init()
    let g:gflags_file_ranker = "~/.vim/autoload/.asconf/ranker/gflags.conf"
    let g:dync_file_ranker = "~/.vim/autoload/.asconf/ranker/switches.conf"
    let g:gflags_file_mixer = "~/.vim/autoload/.asconf/mixer/gflags.conf"
    let g:dync_file_mixer = "~/.vim/autoload/.asconf/mixer/switches.conf"
    let g:last_cursor = [0,0,0,0]
    if filereadable(expand(g:gflags_file_ranker)) == 0 || filereadable(expand(g:dync_file_ranker)) == 0 || filereadable(expand(g:gflags_file_mixer)) == 0 || filereadable(expand(g:dync_file_mixer)) == 0
        return 0
    endif
    let g:is_mixer = system("svn info|grep URL|cut -d' ' -f2|grep mixer")
    let g:is_ranker = system("svn info|grep URL|cut -d' ' -f2|grep ranker")
    if len(g:is_mixer) > 0
        let g:gflags_file = g:gflags_file_mixer
        let g:dync_file = g:dync_file_mixer
        let g:conf_path = "~/.vim/autoload/.asconf/mixer/**"
    elseif len(g:is_ranker) > 0
        let g:gflags_file = g:gflags_file_ranker
        let g:dync_file = g:dync_file_ranker
        let g:conf_path = "~/.vim/autoload/.asconf/ranker/**"
    else
        return 0
    endif

    let on_gflags = '\v--(no)@!(\w+)'
    silent execute "noautocmd vimgrep /" . on_gflags . "/j " . g:gflags_file 
    let g:bd_switch_on_gflags = map(getqflist(), '"FLAGS_".matchlist(v:val.text, on_gflags)[2]')

    let off_gflags = '\v(--no)@<=\w+'
    silent execute "noautocmd vimgrep /" . off_gflags . "/j " . g:gflags_file
    let g:bd_switch_off_gflags = map(getqflist(), 'v:val.text')

    let on_dync = '\v\w+ *: *1'
    silent execute "noautocmd vimgrep /" . on_dync . "/j " . g:dync_file
    let g:bd_switch_on_dync = map(getqflist(), 'matchlist(v:val.text, "\\v\\w+")[0]')

    let off_dync = '\v\w+ *: *0'
    silent execute "noautocmd vimgrep /" . off_dync . "/j " . g:dync_file
    let g:bd_switch_off_dync = map(getqflist(), 'matchlist(v:val.text, "\\v\\w+")[0]')
endfunction

function! bdswitch#highlight()
    if  !exists("g:bdswitchinit")
        call bdswitch#init()
        let g:bdswitchinit = 1
    endif
    if filereadable(expand(g:gflags_file_ranker)) == 0 || filereadable(expand(g:dync_file_ranker)) == 0 || filereadable(expand(g:gflags_file_mixer)) == 0 || filereadable(expand(g:dync_file_mixer)) == 0
        return 0
    endif
    if len(g:is_mixer) == 0 && len(g:is_ranker) == 0
        return 0
    endif

    silent execute 'syn keyword cppFLAGSON ' . join(g:bd_switch_on_gflags, ' ')
    silent execute 'syn keyword cppFLAGSOFF ' . join(g:bd_switch_off_gflags, ' ')
    silent execute 'syn keyword cppDYNCON ' . join(g:bd_switch_on_dync, ' ')
    silent execute 'syn keyword cppDYNCOFF ' . join(g:bd_switch_off_dync, ' ')
    hi def link cppFLAGSON Boolean
    hi def link cppFLAGSOFF Comment
    hi def link cppDYNCON Boolean
    hi def link cppDYNCOFF Comment
endfunction

function! bdswitch#searchconf()
    if !filereadable(expand(g:gflags_file_ranker)) || !filereadable(expand(g:dync_file_ranker)) || !filereadable(expand(g:gflags_file_mixer)) || !filereadable(expand(g:dync_file_mixer))
        return 0
    endif
    if len(g:is_mixer) == 0 && len(g:is_ranker) == 0
        return 0
    endif
    let conf_name='\<'.expand('<cword>').'\>'
    let router_confname_regex = 'REGCFG.*"\(\w\+\)"\s*,\s*\n*\s*&*'.expand('<cword>') . '.*'
    let confname_regex = 'REGCFG.*"\(\w\+\)"\s*'
    silent execute "noautocmd vimgrep /" . conf_name . "/gj ". g:conf_path|redraw
    let confvalue=map(getqflist(), 'v:val.text')
    if len(confvalue) > 0
        copen
    else
        silent execute "noautocmd vimgrep /" . router_confname_regex . "/gj common/** src/** include/** strategy/**"|redraw
        if len(getqflist()) != 1
            if len(getqflist()) > 1
                copen
            endif
            return 0
        endif
        let router_confname = map(getqflist(), 'matchlist(v:val.text, confname_regex)[1]')[0]
        silent execute "noautocmd vimgrep /" . router_confname . "/gj ". g:conf_path|redraw
        if len(getqflist()) < 1
            return 0
        endif
        copen
    endif
endfunction
