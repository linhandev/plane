# plane

```shell
IFS=$'\n'
for folder in `ls`
do
echo
cd "$folder"

var=$(pwd)
basename $(pwd)
currdir="$(basename $PWD)"
echo "$currdir"
ffmpeg -framerate 1 -pattern_type glob -i "*.jpg" ${currdir}.mkv
mv ${currdir}.mkv ..

cd ..
done
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
paddlehub 这个yolo的可视化有bug，一个batch的数据都会用第一张图片之后往上画框
TODO：给这个问题pr
