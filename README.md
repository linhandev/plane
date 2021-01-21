# plane

```shell
IFS=$'\n'

for folder in `ls`
do
echo $folder
cd $folder
ffmpeg -framerate 1 -pattern_type glob -i "*.png" ${folder}.mkv
mv ${folder}.mkv ../../mkv
cd ..
rm $folder -rf
done
```

```shell
dir="Q23502-撤轮挡"
cd $dir
ffmpeg -framerate 2 -pattern_type glob -i "*.png" ${dir}.mkv
mv $dir.mkv ../../mkv
cd ..
rm -rf $dir
```

```shell
mkdir ../mp4
for f in `ls`
do
  cd $f
  mv * ../../mp4
  cd ..
done
```


```shell
for f in `ls`
do
  f=${f%????}
  echo $f
  cp /home/lin/Desktop/data/plane/flg/ann/train/flg/${f}.xml /home/lin/Desktop/data/plane/flg/ann/train/
done
```

python det/f_flg.py -i /home/lin/Desktop/data/plane/frame/img-temp/ -o /home/lin/Desktop/data/plane/frame/xml-temp/ --bs 8 --model model/best_model/



paddlehub 这个yolo的可视化有bug，一个batch的数据都会用第一张图片之后往上画框
TODO：给这个问题pr

注意：
所有数字要补0,否则排序很费劲
一个文件名不要用两个同样的分隔符
