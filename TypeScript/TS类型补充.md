##### 匿名函数的参数类型

​	一般在声明函数时其传参也需要声明参数类型，但匿名函数由于其传入的参数来源已经固定了参数类型，所以可以不进行参数类型的声明。

```typescript
function test1(message: string) {
    
}

const names = ["MXR", "MXRR", "MXRRR"]
 // 其中，item的参数类型可以不进行声明，其来源已经固定为string类型(根据上下文推导而出)
names.forEach((item: string) => {
    console.log(item.split(""))
})
```



##### 对象类型

```typescript
function printPoint(point: {x: number, y: number, z?: number}) {
    console.log(point.x)
    console.log(point.y)
    console.log(point.z)		// 若未传值则打印出undefined
}

printPoint({x: 123, y:321})
printPoint({x: 123, y: 321, z: 666})

export{}
```



##### 联合类型（Union Type）

```typescript
function printID(id: number | string) {
    if (typeof id === 'string') {
        console.log(id.toUpperCase())
    } else {
        console.log(id)
    }
}

// test1和test2等价
function test1(message?: string) {
    console.log(message)
}

function test2(message: string | undefined) {
    console.log(message)
}

printID(123)
printID("hahaha")

```

​	其中，在联合类型过长的情况下可以使用type定义类型别名，使代码整体更美观。

```typescript
type IDType = string | number | boolean
type PointType = {
	x: number,
	y: number,
	z?: number,
}

function test(id: IDType) {
	console.log(id)		// 此处id的类型则为定义中的三种，字面量类型的体现
}

function printPoint(point: PointType) {
	console.log(point)	// 此处同理
}
```



##### 类型断言as

​	当ts无法准确判断类型时可以使用as关键词指明其调用的是哪个元素，如下所示，若不声明id为img标签下的id，其默认会获取浏览器的，导致无法获取到src属性。

​	同时，其也可用于类中的声明，将一个较为宽泛的类型指明其具体的类型是什么。

```typescript
// <img id="MXR"/>
const el： HTMLElement = doucument.getElementById("MXR") as HTMLImageElement

el.src = "url地址"

class Person {}

class Studnet extends Person {
    studying() {}
}

function sayHello(p: Person) {
    (p as Student).studying()
}

const stu = new Student()
sayHello(stu)

//	投机取巧，不建议使用
const message = "123"
const num: number = (message as any) as number
```



##### 非空类型断言

​	当传入值可以为空时若不加以判断则代码无法通过编译。

```typescript
function test(message?: string) {
    console.log(message!.length)
  // 添加判断也可以
  //  if(message) {
  //      console.log(message.length)
  //  }
}

test("qwe")
test("hahaha")
```



##### 可选链

```typescript
type Person = {
    name: string
    friend?: {
        name: string
        age?: number
        grilfriend?: {
            name: string
        }
    }
}

const info: Person = {
    name: "MXR"
    friend: {
    	name: "LBR"
    	grilfriend: {
    		name: "haha"
		}
	}
}

// 属性存在值则取值，不存在不取值且返回undefined
console.log(info.name)
console.log(info.friend?.name)		
console.log(info.friend?.grilfriend?.name)
```



##### ??和!!

​	JS中也可使用此操作符。

```typescript
const message = "MXR"
const flag1 = Boolean(message)		// 此时值为true
const flag2 = !message				// 此时值为false
const flag3 = !!message				// 此时值为true,这个笔记属实有点无语了

let test1: string | null = null		
let test2: string | null = "爷有值"
const content1 = test1 ?? "hahaha"		// 当??左侧值为null或undefined时，取右侧值
const content2 = test2 ?? "hahaha"		// content1值为“hahaha”,content2值为“爷有值”
```



##### 字面量类型

```typescript
// const message = "MXR",默认为字面量类型，与其值保持一致
const message: "MXR" = "MXR"
// const num = 123
const num: 123 = 123

// 字面量类型的意义在于其可以结合联合类型共同使用，可以将多种类型统一为一个类型，并在其中选择一个
type Alignment = 'left' | 'right' | 'center'
let align: Alignment = 'left'
align = 'right'
align = 'center'
// align = 'hahaha'
```



##### 类型缩小

​	就是把参数类型进行了一个细化,若不进行细化则无法确定传入的参数是什么类型，从而将无法调用一些属性和方法。

```typescript
// typeof类型缩小
type IDType = number | string
function printID(id: IDType) {
    if(typeof id === 'sting') {
        console.log(id.toUpperCase)
    }else {
        console.log(id)
    }
}

// 平等类型缩小,使用if和swich都可以达到效果
type Direction = "top" | "buttom" | "left" | "right"
function printDirection(direction: Direction) {
    console.log(direction)			// 此时direction的类型为Direction类型
    if(direction === 'left') {
        console.log(direction)		// 此时direction的类型一定为left
    }else if(direction === 'right') {
        console.log(direction)
    }
}

// instanceof,判断xx是不是某一类型  且instanceof出来的是实例可直接调用方法
type Time: string | Date 
function printTime(time: Time) {
    if(time instanceof Date) {
        console.log(time.toUTCString())
    }else {
        console.log(time)
    }
}

class Student {
    studying() {}
}

class Teacher {
    teaching() {}
}

function work(p: Student | Teacher) {
    if(p instanceof Student) {
        p.studying()
    }else {
        p.teaching()
    }
}

// in
type Fish = {
    swimming: () => void			// 函数类型，此处表明该函数没有返回值
}

type Dog = {
    running: () => void
}

function walk(animal: Fish | Dog) {
    if('swimming' in animal) {
        animal.swimming()
    }else {
        animal.running()
    }
}

const fish: Fish = {
    swimming() {
        console.log("swimming")
    }
}
```

