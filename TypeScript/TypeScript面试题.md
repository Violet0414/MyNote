#### TypeScript主要特点

​	跨平台，具备ES6的大部分特性，可进行DOM操作，可选的静态类型。



#### 使用TypeScript的好处

​	其能够使得开发更为规范，更具表现力，限制变量的类型从而使得语法混乱更少。

​	在编译前就能捕获一些逻辑上的错误，更为容易调试。

​	静态类型也使得相较于JS的动态类型更具阅读性。



#### TS的原始数据类型

​	number, string, boolean, void, null, undefined。其中void类型只能赋值给其null或undefined。



#### 泛型

​	在定义接口，类，函数时不预先指定具体的类型，而在使用时再去指定类型。

​	当需要传入任意类型的值时，同时需要传入什么类型的值，返回就是什么类型，此时就可以使用泛型。

```typescript
function createArray1(length: any, value: any): Array<any> {
    let result: any = [];
    for (let i = 0; i < length; i++) {
        result[i] = value;
    }
    return result;
}

let result = createArray1(3, 'x');
console.log(result);

// 最傻的写法：每种类型都得定义一种函数
function createArray2(length: number, value: string): Array<string> {
    let result: Array<string> = [];
    for (let i = 0; i < length; i++) {
        result[i] = value;
    }
    return result;
}

function createArray3(length: number, value: number): Array<number> {
    let result: Array<number> = [];
    for (let i = 0; i < length; i++) {
        result[i] = value;
    }
    return result;
}

// 或者使用函数重载，写法有点麻烦
function createArray4(length: number, value: number): Array<number>
function createArray4(length: number, value: string): Array<string>
function createArray4(length: number, value: any): Array<any> {
    let result: Array<number> = [];
    for (let i = 0; i < length; i++) {
        result[i] = value;
    }
    return result;
}
createArray4(6, '666');

// 使用泛型
// 有关联的地方都改成 <T>
function createArray<T>(length: number, value: T): Array<T> {
    let result: T[] = [];
    for (let i = 0; i < length; i++) {
        result[i] = value;
    }
    return result;
}
// 使用的时候再指定类型
let result = createArray<string>(3, 'x');
// 也可以不指定类型，TS 会自动类型推导
let result2 = createArray(3, 'x');
console.log(result);
```



#### 可索引类型接口

​	用于约束数组和对象

```typescript
/ 数字索引——约束数组
// index 是随便取的名字，可以任意取名
// 只要 index 的类型是 number，那么值的类型必须是 string
interface StringArray {
  // key 的类型为 number ，一般都代表是数组
  // 限制 value 的类型为 string
  [index:number]:string
}
let arr:StringArray = ['aaa','bbb'];
console.log(arr);


// 字符串索引——约束对象
// 只要 index 的类型是 string，那么值的类型必须是 string
interface StringObject {
  // key 的类型为 string ，一般都代表是对象
  // 限制 value 的类型为 string
  [index:string]:string
}

let obj:StringObject = {name:'ccc'};

```

