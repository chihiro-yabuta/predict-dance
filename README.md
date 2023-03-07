## What is predict-dance ?
***predict-dance*** is enable to decide which videos belong to ***[elegant, dance, other]***.

<table>
  <tr><th>\</th><th>resize</th><th>gray</th><th>binary</th><th>limit</th></tr>
  <tr>
    <th>elegant</th>
    <td><img src="src/git/ja_resize.gif" width=159></td>
    <td><img src="src/git/ja_gray.gif" width=159></td>
    <td><img src="src/git/ja_bin.gif" width=159></td>
    <td><img src="src/git/ja_limit.gif" width=159></td>
  </tr>
  <tr>
    <th>dance</th>
    <td><img src="src/git/aito_resize.gif" width=159></td>
    <td><img src="src/git/aito_gray.gif" width=159></td>
    <td><img src="src/git/aito_bin.gif" width=159></td>
    <td><img src="src/git/aito_limit.gif" width=159></td>
  </tr>
  <tr>
    <th>other</th>
    <td><img src="src/git/exer_resize.gif" width=159></td>
    <td><img src="src/git/exer_gray.gif" width=159></td>
    <td><img src="src/git/exer_bin.gif" width=159></td>
    <td><img src="src/git/exer_limit.gif" width=159></td>
  </tr>
</table>
you will get result of video prediction.

![flowchart](src/git/result.png)

## How do I start ?
plz install docker desktop first.<br>
after that, you can start to type default Makefile command.
```
make
```
I have already prepared commands.<br>
plz check Makefile.<br>
you can develop in docker container as well.<br>

when you finish to load, you probably can see directories.<br>
I show you tree to understand easily.
<pre>
.
├── archive
│   ├── *.mp4 (pred video)
├── out
│   ├── video (stock visible pred)
│   │   ├── cam
│   │   │   ├── *.mp4
│   │   ├── edited
│   │   │   ├── *.mp4
│   │   ├── removed
│   │   │   ├── *.mp4
│   └── flow (stock visible pred detail)
│   │   ├── ...
│   └── src (stock data)
│   │   ├── edited
│   │   │   ├── *.pkl
│   │   ├── removed
│   │   │   ├── *.pkl
│   ├── model (stock overall data)
│   │   ├── *.pkl
├── src
│   ├── ...
├── test
│   ├── *.mp4 (test video)
└── video
    ├── *.mp4 (study video)
</pre>
you can prepare data to predict to type this command.
```
make dp
```
you can execute prediction to type these commands.
```
python main.py
```
```
python test.py
```

you can modify code everything. plz read my code and improve better!<br>
I show you recent nn and old repos for help.
[pre1](https://github.com/jasmine-jp/predict_dance)
[pre2](https://github.com/jasmine-jp/predict_dance2)
## flow chart
![flowchart](src/git/flowchart.png)
