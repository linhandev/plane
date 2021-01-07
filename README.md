# plane

```shell
var=$(pwd)
basename $(pwd)
currdir="$(basename $PWD)"
echo "$currdir"
ffmpeg -framerate 1 -pattern_type glob -i "*.png" ${currdir}.mp4
```
