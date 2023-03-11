#!/bin/bash

rm  file_dt_temp.txt
tt=`date -d "7 day ago" +%Y%m%d`

hdfs dfs -ls /user/hadoop-hmart-waimaiad/ | awk -F "/" '{print $NF}' | while read line
do
    #echo $line
    # 删除7天之前的数据
    if [ $line -le $tt ];then
        echo "$line"  >> /home/hdfs/file_dt_temp.txt
        echo "hdfs  dfs -rm -r  hdfs://xxxxxxxxx/ranger/audit/hdfs/$line "
        echo "hdfs  dfs -rm -r  hdfs://xxxxxxxxx/ranger/audit/yarn/$line "
#        hdfs  dfs -rm -r  hdfs://xxxxxxxxx/ranger/audit/hdfs/$line
#        hdfs  dfs -rm -r  hdfs://xxxxxxxxx/ranger/audit/yarn/$line
    fi
done
