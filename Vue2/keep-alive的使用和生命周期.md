**keep-alive的使用**：

**keep-alive**是 Vue 内置的一个组件，可以使被包含的组件保留状态，或避免重新渲染。
在Vue 2.1.0 版本之后，keep-alive新加入了两个属性: include(包含的组件缓存) 与 exclude(排除的组件不缓存，优先级大于include) 。



**使用方法：**

```html
<keep-alive include='include_components' exclude='exclude_components'>
  <component>
    <!-- 该组件是否缓存取决于include和exclude属性 -->
  </component>
</keep-alive>
```

**参数解释**：

include - 字符串或正则表达式，只有名称匹配的组件会被缓存
exclude - 字符串或正则表达式，任何名称匹配的组件都不会被缓存
include 和 exclude 的属性允许组件有条件地缓存。二者都可以用“，”分隔字符串、正则表达式、数组。当使用正则或者是数组时，要记得使用v-bind 。

**例子：**

```html
<!-- 逗号分隔字符串，只有组件a与b被缓存。 -->
<keep-alive include="a,b">
  <component></component>
</keep-alive>
 
<!-- 正则表达式 (需要使用 v-bind，符合匹配规则的都会被缓存) -->
<keep-alive :include="/a|b/">
  <component></component>
</keep-alive>
 
<!-- Array (需要使用 v-bind，被包含的都会被缓存) -->
<keep-alive :include="['a', 'b']">
  <component></component>
</keep-alive>
```



**生命周期与keepalive的关系：**

​    <keep-alive>包裹的动态组件会被缓存，它是一个抽象组件，它自身不会渲染一个dom元素，也不会出现在父组件链中。当组件在 <keep-alive> 内被切换，它的 activated 和 deactivated 这两个生命周期钩子函数将会被对应执行。

​    如<keep-alive>包裹两个组件：组件A和组件B。当第一次切换到组件A时，组件A的created和activated生命周期函数都会被执行，这时通过点击事件改变组件A的文字的颜色，在切换到组件B，这时组件A的deactivated的生命周期函数会被触发；在切换回组件A，组件A的activated生命周期函数会被触发，但是它的created生命周期函数不会被触发了，而且A组件的文字颜色也是我们之前设置过的。
