#### 接口

```typescript
// 使用type别名声明对象类型
// type InfoType = {name: string, age: number, height: string}

// 另一种方式：接口interface,接口命名规范在名称前加个I
interface IInfoType {
  readonly name: string,        // 只读属性
  age: number,
  height: number,
  friend?: {
    name: string
  }
}

const info: IInfoType = {
  name: "MXR",
  age: 22,
  height: 1.73,
}

// 此处无法修改只读属性name
// info.name = "xxx"   
info.age = 18
```



#### 索引类型

​	此处的key索引好像有问题

```typescript
// 接口的索引类型
interface IIndexLanguage {
  [indxe: number]: string
}

// 此时通过接口的类型定义限制对象内的赋值格式
const frontLanguage: IIndexLanguage = {
  0: 'HTML',
  1: 'CSS',
  2: 'JavaScript',
  3: 'Vue',
  // 4: 123
}

interface ILanguageYear {
  [name: string]: number
}

const languageYear: ILanguageYear = {
  "C": 1972,
  "Java": 1995,
  "JavaScript": 1996,
  "TypeSript": 2014,
  123: 123
}
```



#### 通过接口定义函数类型

```typescript
// 阿巴阿巴
```

