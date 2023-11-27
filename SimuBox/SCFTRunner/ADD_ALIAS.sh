#!/bin/bash

# 要添加的alias语句
alias_list=(
    'current_user=$(whoami)'
    'alias prep="cp /home/${current_user}/SCFTRunner/Scripts/* ./"'
    'alias babcb="cp /home/${current_user}/SCFTRunner/JSON/BABCB.json ./input.json"'
    'alias abcabc="cp /home/${current_user}/SCFTRunner/JSON/ABC_ABC.json ./input.json"'
    'alias prep_babcb="prep;babcb"'
    'alias prep_abcabc="prep;abcabc"'
)

for element in "${alias_list[@]}"; do
    # 检查.bashrc是否已经包含该alias，如果不存在则添加
    if ! grep -q "$element" ~/.bashrc; then
        echo "$element" >> ~/.bashrc
        echo "$element added to .bashrc"
    else
        echo "$element already exists in .bashrc"
    fi
done

# 使更改生效
source ~/.bashrc
