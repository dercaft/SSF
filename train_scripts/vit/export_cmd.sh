#!/bin/bash
# 创建一个空的脚本，用于存放所有命令
echo "" > all_commands.sh
# 遍历vtab文件夹下的所有子文件夹
for dir in vtab/*/; do
    # 从文件路径中提取文件夹名称
    dir_name=$(basename "$dir")
    # 创建一个空字符串，用于存放由train_ssf.sh脚本中读取的命令
    command=""
    while IFS= read -r line
    do
        # 将train_ssf.sh脚本中的每一行命令添加到字符串中
        command+="$line && "
    done < "$dir/train_ssf.sh"
    # 移除字符串末尾的" && "
    command=${command%&&}
    # 将完整脚本的命令（包括输出重定向）写入all_commands.sh
    echo "($command) > param/${dir_name}.txt" >> all_commands.sh
done