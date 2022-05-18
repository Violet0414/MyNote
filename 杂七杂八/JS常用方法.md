#### 类型判断

##### 	typeof():

​	可判断基本类型，但无法区分Object，Array，Null。

```js
console.log(typeof 1);               // number
console.log(typeof true);            // boolean
console.log(typeof 'mc');            // string
console.log(typeof Symbol)           // function
console.log(typeof function(){});    // function
console.log(typeof console.log());   // function
console.log(typeof []);              // object 
console.log(typeof {});              // object
console.log(typeof null);            // object
console.log(typeof undefined);       // undefined

```



##### 	instanceof():

​	可区分Object，Array，Null，但无法判断Number，String，Boolean。

```js
console.log(1 instanceof Number);                    // false
console.log(true instanceof Boolean);                // false 
console.log('str' instanceof String);                // false  
console.log([] instanceof Array);                    // true
console.log(function(){} instanceof Function);       // true
console.log({} instanceof Object);                   // true
```



##### replace():

​	两个参数，第一个是正则，第二个是替换字符，符合正则表达式的字符将会被该参数替换。

```js
let str = 'Hello, world!'
str = str.replace(/,/g, '')
console.log(str)	// 输出Hello world!
```



##### pop()，改变原数组

```js
var arr = [1,2,3]
var newarr = arr.pop()
console.log(arr);		// [1,2]
console.log(newarr);	// 3
```



##### shift()，改变原数组

```js
var arr = [1,2,3]
var newarr = arr.shift()
console.log(arr);		// [2,3]
console.log(newarr);	// 1
```



##### unshift()，改变原数组

```js
var arr = [1,2,3]
var newarr = arr.unshift(5)
console.log(arr);			// [5,1,2,3]
console.log(newarr);		// 4
```



##### push()，改变原数组

```js
 var arr = [1,2,3]
 var newarr = arr.push(4)
 console.log(arr);  	// [1,2,3,4]
 console.log(newarr);	// 4
```



##### splice()，改变原数组

```js
var arr = [1,2,3]
var newarr = arr.splice(2,0,5)
console.log(arr);			// [1,2,5,3]
console.log(newarr);		// []

var arr = [1,2,3]
var newarr = arr.splice(0,1)
console.log(arr);			// [2,3]
console.log(newarr);		// [1]

var arr=[1,2,3]
var newarr = arr.splice(0,1,4)
console.log(arr);			// [4,2,3]
console.log(newarr);		// [1]
```



##### concat()，不改变原数组

```js
var arr = [1,2,3,2,9]
var newarr = arr.concat(1,2)
console.log(arr);			// [1,2,3,2,9]
console.log(newarr);		// [1,2,3,2,9,1,2]

var arr2=[1,0]
var newarr2 = arr.concat(arr2)
console.log(newarr2);		// [1,2,3,2,9,1,0]
```



##### slice()，不改变原数组

```js
var arr = [1,2,3,2,9]
var newarr = arr.slice(1,2)
console.log(arr);			// [1,2,3,2,9]
console.log(newarr);		// [2]
```



##### indexOf()

```js
var arr = [1,2,3]
var newarr = arr.indexOf(2)
console.log(newarr);		// 1，若没查到值则返回-1
```



##### every()

```js
var arr = [6,2,3,2,9]
var newarr1 = arr.every((item,index) => {
    return item > 2
})
var newarr2 = arr.every((item,index) => {
    return item > 1
})
console.log(newarr1);   		// false
console.log(newarr2);   		// true
```



##### set()

```js
var arr=[1,2,3,2,9]
var newarr = [...new Set(arr)]
console.log(newarr);		// [1, 2, 3, 9]

var a = new Set([1, 2, 3, 4]);
console.log(a);				//Set(4) {1, 2, 3, 4}
a.has(2);					// true
a.add(5);
console.log(a)				//Set(5) {1, 2, 3, 4, 5}
a.delete(3);
console.log(a);				//Set(4) {1, 2, 4, 5}
a.clear();
console.log(a); 			// Set(0) {size: 0}

var arr=[1,2,3,2,9]
var newarr = [...new Set(arr)]
console.log(newarr);//[1, 2, 3, 9]
```

