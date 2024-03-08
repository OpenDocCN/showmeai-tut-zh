<!--
    需要填充的占位符：
    
    README.md
    
        ShowMeAI 教程合集：文档中文名
        {nameEn}：文档英文名
        {urlEn}：文档原始链接
        smai：域名前缀
        飞龙：负责人名称
        wizardforcel：负责人 Github 用户名
        562826179：负责人 QQ
        showmeai-tut-zh：ApacheCN 的 Github 仓库名称
        showmeai-tut-zh：DockerHub 仓库名称
        showmeai-tut-zh：PYPI 包名称
        showmeai-tut-zh：NPM 包名称
    
    CNAME
    
        smai：域名前缀

    index.html
    
        ShowMeAI 教程合集：文档中文名
        #1E90FF：显示颜色
        showmeai-tut-zh：ApacheCN 的 Github 仓库名称

    asset/docsify-flygon-footer.js
    
        showmeai-tut-zh：ApacheCN 的 Github 仓库名称
-->

# ShowMeAI 教程合集

> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)
> 
> 真相一旦入眼，你就再也无法视而不见。——《黑客帝国》

* [在线阅读](https://smai.flygon.net)

## 下载

### Docker

```
docker pull apachecn0/showmeai-tut-zh
docker run -tid -p <port>:80 apachecn0/showmeai-tut-zh
# 访问 http://localhost:{port} 查看文档
```

### NPM

```
npm install -g showmeai-tut-zh
showmeai-tut-zh <port>
# 访问 http://localhost:{port} 查看文档
```

