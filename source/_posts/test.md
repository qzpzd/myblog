---
title: butterfly教程
date: 2023-07-27 15:08:29
type: "tags"
tags:
   -butterfly
   -教程
keywords: 'butterfly'
description: 对butterfly主题的简单配置介绍
cover: https://file.crazywong.com/gh/jerryc127/CDN/img/butterfly-docs-01-cover.png
top_img: 
---
# 引用块
源代码
{% codeblock [引用块]  %}
{% blockquote author, - source [link] [source_link_title] %}
content  
{% endblockquote %}
{% endcodeblock %}

{% blockquote author, - source [link] [source_link_title] %}
content  
{% endblockquote %}

源代码
{% codeblock [引用块]  %}

>引用内容
>>引用内容
>>>yy内容

{% endcodeblock %}

>引用内容
>>引用内容
>>>yy内容

# 分割线
源代码
{% codeblock [分割线]  %}

---

{% endcodeblock %}

---


# 代码块
源代码
{% codeblock [代码块]  %}

{% codeblock [title] [lang:language] [url] [link text] %}  
code snippet  
{% endcodeblock %}  

{% endcodeblock %}

{% codeblock [title] [lang:language] [url] [link text] %}  
code snippet  
{% endcodeblock %}  

# 引用文章链接
源代码
{% codeblock [链接块]  %}

{% post_link hello-world 'hello-world' %} 

{% endcodeblock %}

{% post_link hello-world 'hello-world' %}

# 选项卡
源代码
{% codeblock [选项卡]  %}

{% tabs 标签, 1 %} 
<!-- tab -->
**选项卡 1** 
<!-- endtab -->
<!-- tab -->
**选项卡 2**
<!-- endtab -->
<!-- tab 标签三 -->
**选项卡 3** , 名字为 `TAB三`
<!-- endtab -->
{% endtabs %} 

{% endcodeblock %}


{% tabs 标签, 1 %} 
<!-- tab -->
**选项卡 1** 
<!-- endtab -->
<!-- tab -->
**选项卡 2**
<!-- endtab -->
<!-- tab 标签三 -->
**选项卡 3** , 名字为 `TAB三`
<!-- endtab -->
{% endtabs %}


源代码
{% codeblock [选项卡]  %}

<div class="gallery-group-main">
{% galleryGroup '壁纸' '收藏的一些壁纸' '/Gallery/wallpaper/' https://api.aqcoder.cntoday %}
</div>

{% endcodeblock %}

<div class="gallery-group-main">
{% galleryGroup '壁纸' '收藏的一些壁纸' '/gallery/wallpaper/' https://api.aqcoder.cntoday %}
</div>


# 提示块
源代码
{% codeblock [提示块]  %}

{% note default simple %}
default 提示块
{% endnote %}

{% note primary simple %}
primary 提示块
{% endnote %}

{% note success simple %}
success 提示块
{% endnote %}

{% note info simple %}
info 提示块
{% endnote %}

{% note warning simple %}
warning 提示块
{% endnote %}

{% note danger simple %}
danger 提示块
{% endnote %}

{% endcodeblock %}


{% note default simple %}
default 提示块
{% endnote %}

{% note primary simple %}
primary 提示块
{% endnote %}

{% note success simple %}
success 提示块
{% endnote %}

{% note info simple %}
info 提示块
{% endnote %}

{% note warning simple %}
warning 提示块
{% endnote %}

{% note danger simple %}
danger 提示块
{% endnote %}







