#!/usr/bin/env bash
# usage:  ./run_collector.sh

PYTHON=python3
SCRIPT="misc/data_collect.py"
ARGS="--save-path ~/code/data/ --save-num 50000"

LOG_DIR=~/code/data/logs
mkdir -p "$LOG_DIR"

cnt=0
while :; do
    ((cnt++))
    log="$LOG_DIR/collect_$(date +%F_%H-%M-%S)_$$.log"
    echo "[$(date)] 第${cnt}次启动采集 —— 日志: $log"

    # 真正启动采集
    $PYTHON $SCRIPT $ARGS >>"$log" 2>&1
    exit_code=$?

    # 正常跑满 50000 帧 -> 脚本自己 exit(0)
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date)] 采集完成，正常退出。"
        break
    fi

    # 130=Ctrl-C，用户主动停就退出
    if [[ $exit_code -eq 130 ]]; then
        echo "[$(date)] 用户中断，停止重启。"
        break
    fi

    # 其它情况 -> 认为 core/异常
    echo "[$(date)] 检测到异常退出码 $exit_code，3 秒后重启..."
    sleep 3
done