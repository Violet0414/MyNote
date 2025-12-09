##### 函数类型

```typescript
// 将函数作为参数传入
function foo() {}
function bar(fn: () => void) {
    fn()
}

// 也可以使用type声明
type FnType = () => void
function test(fn: FnType) {
	fn()
}

bar(foo)
test(foo)

// 定义常量时，编写函数的类型，声明传参类型以及返回值类型
type AddType = (num1: number, num2: number) => number	
// 此处返回值为void时返回值类型为任何都可 
const add: AddType = (num1: number, num2: number) => {
    return num1 + num2
}

// ======================== 案例 ======================
function calc(n1: number, n2: number, fn: (num1: number, num2: number) => number) {
    return fn(n1, n2)
}

// 此处传过来的参数可以不定义类型，因为上述代码已经定义了
calc(10, 20, function(test1, test2) {
    return test1 + test2				// 输出结果为30
})

calc(10, 20, function(test1, test2) {
    return test1 * test2				// 输出结果为200
})
```





##### 函数的可选类型

```typescript
// 可选类型必须写在必选类型后面
// y此时相当于 y: undefined | number,但如果写成左边那种，此时参数就必传
function foo(x: number, y?: number) {
    return x + y
}

foo(20, 30)
foo(20)
```



##### 参数默认值

```typescript
// 最好是先写必传参数，然后再写有默认值的参数
// 否则在传值时如果使用默认值则需要在默认值参数位传入undefined
function foo(x: number, y: number = 100) {
    console.log(x, y)
}

function test(x: number = 100, y: number) {
    console.log(x, y)
}

// 此时不传值y将使用默认值100
foo(20)
// 此时必传undefined,x将使用默认值100
test(undefined, 20)
```



##### 函数的剩余参数

```
function sum(...nums: number[])
```



##### 函数的重载

​	函数名相同但参数不同，函数重载后的操作在具体实现函数内进行编写。

​	开发中尽量使用联合类型实现，当联合类型无法实现的情况下再考虑函数重载。

```typescript
// TS中函数重载可以没有函数的具体实现
function add(num1: number, num2: number): number;
function add(num1: string, num2: string): string;

function add(num1: any, num2: any): any {
    return num1 + num2
}

console.log(add(66, 66))
console.log(add('66', '66'))

// 函数重载中实现函数无法被直接调用
// add({name: "MXR"}, {age: 22})

export {}
```

