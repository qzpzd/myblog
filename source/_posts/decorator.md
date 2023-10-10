---
title: decorator
date: 2023-08-10 14:51:35
type: "tags"
tags:
    -python
    -decorator
keywords: 'decorator'
cover: https://s2.loli.net/2023/08/10/qckoIznlQp4Ot69.jpg
---
# 1、@timer:测量执行时间

@timer装饰器可以帮助我们跟踪特定函数的执行时间。通过用这个装饰器包装函数，我可以快速识别瓶颈并优化代码的关键部分。

{% codeblock [测量执行时间]  %}
import time
 
def timer(func):
   def wrapper(*args, **kwargs):
       start_time = time.time()
       result = func(*args, **kwargs)
       end_time = time.time()
       print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
       return result
   return wrapper
@timer
def my_data_processing_function():
   # Your data processing code here

{% endcodeblock %}

# 2.@log_results:日志输出

在运行复杂的数据分析时，跟踪每个函数的输出变得至关重要。@log_results装饰器可以帮助我们记录函数的结果，以便于调试和监控

{% codeblock [日志输出]  %}
def log_results(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        with open("results.log", "a") as log_file:
            log_file.write(f"{func.__name__} - Result: {result}\n")
        return result
 
    return wrapper
 @log_results
def calculate_metrics(data):
   # Your metric calculation code here

{% endcodeblock %}

# 3.@suppress_errors:优雅的错误处理

@suppress_errors装饰器可以优雅地处理异常并继续执行,可以避免隐藏严重错误，还可以进行错误的详细输出，便于调试.

{% codeblock [错误处理]  %}
def suppress_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
 
    return wrapper
@suppress_errors
def preprocess_data(data):
   # Your data preprocessing code here

{% endcodeblock %}

# 4.@debug:调试变得更容易

调试复杂的代码可能非常耗时。@debug装饰器可以打印函数的输入参数和它们的值，以便于调试:

{% codeblock [调试]  %}
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Debugging {func.__name__} - args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
 
    return wrapper
@debug
def complex_data_processing(data, threshold=0.5):
   # Your complex data processing code here

{% endcodeblock %}


{% blockquote 数据STUDIO, - source [https://mp.weixin.qq.com/s/JFaH_GqOFMARnzyWqV2TFQ] [10个简单但很有用的Python装饰器] %}
>1.@timer:测量执行时间
>2.@log_results:日志输出
>3.@suppress_errors:优雅的错误处理
>4.@debug:调试变得更容易
{% endblockquote %}
