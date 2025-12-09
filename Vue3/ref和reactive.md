#### **简介**

  ref和reactive是Vue3新增的两个api，用于响应式数据的声明和更新，做的精准检测跟踪js中所有的数据类型变动，并且能够达到vnode的对比后真实dom的渲染。



| 比较维度(kin注)         |     ref      |    reactive     |
| :---------------------- | :----------: | :-------------: |
| 是否响应式对象 JS Proxy |      是      |       是        |
| 创建的数据类型          | 任何数据类型 |   对象或数组    |
| 是否需要.value属性      |      是      |       否        |
| 复杂的类型标注          | ref 这个类型 | interface自定义 |
| 隐式推导类型            |      是      |       是        |
| dom的更新               |     异步     |      异步       |
| 是否深层次响应          |  默认深层次  |   默认深层次    |



```vue
<template>
  <button @click="increment">
    {{ count }}
  </button>
</template>

<script setup>
import { ref } from 'vue'
 
const count = ref(0)
 
function increment() {
  count.value++
}
</script>
```



#### **ref()声明数据为何需要使用.value**

​	在Vue2中变量在data()函数内进行声明作统一管理，而在Vue3中使用**ref()**可随时进行响应式变量的声明，就导致声明的响应式数据可能不再是一个对象。

​	**Vue2数据代理：**

​		通过**Object.defineProperty()**实现，该方法会在对象上定义一个新属性，或者修改一个对象的现有属性，之后返回此对象，以此实现数据的响应式。

​	**Vue3数据代理：**

​		通过**Proxy**对象进行数据代理，该对象用于创建一个对象代理，从而实现对数据基本操作的拦截和自定义（如查找，赋值，枚举，函数调用）。

​	由此可以看出，**Proxy代理方式本质仍是对象服务**，但在Vue3中使用该方式也不能进行普通数据的代理，所以在调用ref()时在创建Proxy对象的同时，在其对象上添加了一个value属性，该属性值则是定义的内容。**在其值改变时则是通过监听Proxy的数据劫持来进行响应式数据处理**，而在模板中使用时Vue则会默认调用对应的.value属性，从而完成对数据的操作使用。





#### **ref语法糖**

​	该语法糖**在ref前加上$符号**，可在修改或使用声明数据时不加.value。

该功能需手动开启

```ts
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig({
  plugins: [
    vue({
      refTransform: true // 开启ref转换
    })
  ]
})
```

在.vue文件中使用

```vue
<template>
    <div>{{count}}</div>
    <button @click="add">click me</button>
</template>

<script setup>
    let count = $ref(1)
    const add = () => {
        count++
    }
</script>
```

