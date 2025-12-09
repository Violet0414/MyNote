### vm.$el

获取Vue实例关联的DOM元素；

### vm.$data

获取Vue实例的data选项（对象）

### vm.$options

获取Vue实例的自定义属性（如vm.$options.methods,获取Vue实例的自定义属性methods）

### vm.$refs

获取页面中所有含有ref属性的DOM元素（如vm.$refs.hello，获取页面中含有属性ref = “hello”的DOM元素，如果有多个元素，那么只返回最后一个）

### Js代码

```js
var app   = new Vue({    
        el:"#container",    
        data:{    
        	msg:"hello,2018!"    
        },    
        address:"长安西路" 
})    
```

### console.log(app.$el);

返回Vue实例的关联DOM元素，在这里是#container



### console.log(app.$data);

返回Vue实例的数据对象data，在这里就是对象{msg：”hello，2018“}



### console.log(app.$options.address);

返回Vue实例的自定义属性address，在这里是自定义属性address



### console.log(app.$refs.hello)

返回含有属性ref = hello的DOM元素（如果多个元素都含有这样的属性，只返回最后一个）

<h3 ref = "hello">呵呵 1{{msg}}</h3>