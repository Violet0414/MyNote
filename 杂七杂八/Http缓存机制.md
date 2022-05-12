## Http Cache:

**长久缓存复用重复不变的资源文件是性能优化的重要组成部分。**

配置服务器**响应头**来告诉浏览器是否应该缓存资源、是否强制校检缓存、缓存多长时间。

浏览器**首次请求根据响应头**是否应该读取缓存、**缓存过期** 是否发送请求头验证是否可用 还是重新获取资源。

![](E:\笔记\笔记图片\http缓存.png)



#### 明确禁止缓存：

**设置响应头：**

  **Cache-Control: no-store** 或 **Cache-Control: no-cache, no-store, must-revalidate**

```js
strategy['no-store'](req, res, filePath, stat);
```



**Cache-Control: public** 

  表示一些中间代理、CDN等可以缓存资源，即便是带有一些敏感 HTTP 验证身份信息甚至响应状态代码通常无法缓存的也可以缓存。通常 public 是非必须的，因为响应头 max-age 信息已经明确告知可以缓存了。

**Cache-Control: private** 

  明确告知此资源只能单个用户可以缓存，其他中间代理不能缓存。原始发起的浏览器可以缓存，中间代理不能缓存。例如：百度搜索时，特定搜索信息只能被发起请求的浏览器缓存。



#### **缓存过期策略**

一般缓存机制只作用于 get 请求。



#### **三种方式设置服务器告知浏览器缓存过期时间**

  设置响应头（注意浏览器有自己的缓存替换策略，即便资源过期，不一定被浏览器删除。同样资源未过期，可能由于缓存空间不足而被其他网页新的缓存资源所替换而被删除。）



1.**设置 Cache-Control: max-age=1000**   //响应头中的 Date 经过 1000s 过期。

2.**设置 Expires**   //此时间与本地时间(响应头中的 Date )对比，小于本地时间表示过期，由于本地时钟与服务器时钟无法保持一致，导致比较不精确。

3.如果以上均未设置，却设置了 Last-Modified ，浏览器隐式的**设置资源过期时间为 (Date - Last-Modified) * 10% 缓存过期时间。**



#### **两种方式校验资源过期**

设置请求头：

1、**If-None-Match** 如果缓存资源过期，浏览器发起请求会自动把原来缓存响应头里的 ETag 值设置为请求头 If-None-Match 的值发送给服务器用于比较。一般设置为文件的 hash 码或其他标识能够精确判断文件是否被更新，为强校验。
2、**If-Modified-Since** 同样对应缓存响应头里的 Last-Modified 的值。此值可能取得 ctime 的值，该值可能被修改但文件内容未变，导致对比不准确，为弱校验。

## 强制校验缓存

  有时我们既想享受缓存带来的性能优势，可有时又不确认资源内容的更新频度或是其他资源的入口，我们想此服务器资源一旦更新能立马更新浏览器的缓存，这时我们可以设置

**Cache-Control: no-cache**





## 示例：

```js
let http = require('http');
let url = require('url');
let path = require('path');
let fs = require('fs');
let mime = require('mime');// 非 node 内核包，需 npm install
let crypto = require('crypto');

// 缓存策略
const strategy = {
    'nothing': (req, res, filePath) => {
        fs.createReadStream(filePath).pipe(res);
    },
    'no-store': (req, res, filePath, stat) => {
        // 禁止缓存
        res.setHeader('Cache-Control', 'no-store');
        // res.setHeader('Cache-Control', ['no-cache', 'no-store', 'must-revalidate']);
        // res.setHeader('Expires', new Date(Date.now() + 30 * 1000).toUTCString());
        // res.setHeader('Last-Modified', stat.ctime.toGMTString());

        fs.createReadStream(filePath).pipe(res);
    },
    'no-cache': (req, res, filePath, stat) => {
        // 强制确认缓存
        // res.setHeader('Cache-Control', 'no-cache');
        strategy['cache'](req, res, filePath, stat, true);
        // fs.createReadStream(filePath).pipe(res);
    },
    'cache': async (req, res, filePath, stat, revalidate) => {
        let ifNoneMatch = req.headers['if-none-match'];
        let ifModifiedSince = req.headers['if-modified-since'];
        let LastModified = stat.ctime.toGMTString();
        let maxAge = 30;

        let etag = await new Promise((resolve, reject) => {
            // 生成文件 hash
            let out = fs.createReadStream(filePath);
            let md5 = crypto.createHash('md5');
            out.on('data', function (data) {
                md5.update(data)
            });
            out.on('end', function () {
                resolve( md5.digest('hex') );
            });
        });
        console.log(etag);
        if (ifNoneMatch) {
            if (ifNoneMatch == etag) {
                console.log('304');
                // res.setHeader('Cache-Control', 'max-age=' + maxAge);
                // res.setHeader('Age', 0);
                res.writeHead('304');
                res.end();
            } else {
                // 设置缓存寿命
                res.setHeader('Cache-Control', 'max-age=' + maxAge);
                res.setHeader('Etag', etag);
                fs.createReadStream(filePath).pipe(res);
            }
        }
        /*else if ( ifModifiedSince ) {
            if (ifModifiedSince == LastModified) {
                res.writeHead('304');
                res.end();
            } else {
                res.setHeader('Last-Modified', stat.ctime.toGMTString());
                fs.createReadStream(filePath).pipe(res);
            }
        }*/
        else {
            // 设置缓存寿命
            // console.log('首次响应！');
            res.setHeader('Cache-Control', 'max-age=' + maxAge);
            res.setHeader('Etag', etag);
            // res.setHeader('Last-Modified', stat.ctime.toGMTString());

            revalidate && res.setHeader('Cache-Control', [
                'max-age=' + maxAge,
                'no-cache'
            ]);
            fs.createReadStream(filePath).pipe(res);
        }
    }

};

http.createServer((req, res) => {
    console.log( new Date().toLocaleTimeString() + '：收到请求')
    let { pathname } = url.parse(req.url, true);
    let filePath = path.join(__dirname, pathname);
    // console.log(filePath);
    fs.stat(filePath, (err, stat) => {
        if (err) {
            res.setHeader('Content-Type', 'text/html');
            res.setHeader('404', 'Not Found');
            res.end('404 Not Found');
        } else {
            res.setHeader('Content-Type', mime.getType(filePath));

            // strategy['no-cache'](req, res, filePath, stat);
            // strategy['no-store'](req, res, filePath, stat);
            strategy['cache'](req, res, filePath, stat);
            // strategy['nothing'](req, res, filePath, stat);
        }
    });
})
.on('clientError', (err, socket) => {
    socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
})
.listen(8080);



```





