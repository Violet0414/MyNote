### Vue的传值方式:

#### props:

```vue
<template>
	<div>
        <children :msg = "test"></children>
    </div>
</template>

<script>
	export default {
        data() {
            return {
                test: '我是测试'
            }
        }
    }
</script>

// 子组件
<template>
	<p>{{msg}}</p>
</template>

<script>
	export default {
        // props: ["msg"]
        props {
            msg: {
                type: String,                 //可指定接收类型，如:Array.
                default:"this is default"     //可设置默认值
            }
        }
    }
</script>

```



#### $emit:

```vue
<template>
	<div>
        <button @click = "send">发送事件</button>
    </div>
</template>

<script>
	export default {
        data() {
            return {
                val: '我是子组件传来的数据'
            }
        }

        methods: {
            send() {
                this.$emit('sendData', this.val)
            }
        }
    }
</script>

// 父组件
<template>
	<div>
        <children @sendData = "getSend">发送事件</button>
        <p>{{test}}</p>
    </div>
</template>

<script>
	export default {
        data() {
            return {
                test: '',
            }
        }

        methods: {
            getSend(val) {
                this.test = val
            }
        }
    }
</script>
```



#### provide:

```vue
<script>
	export default {
        provide: {
            name: "MXR",
            age: 22,
        }
    }
</script>

// 孙组件
<template>
	<div>
        <p>{{name}} --- {{age}}</p>
    </div>
</template>

<script>
	export default {
        inject: ["name", "age"]
    }
</script>
```



#### emitter:

  单独创建一个js文件用于引入事件总线，在需要发送事件的组件进行引入。

```vue
<template>
	<div>
        <button @click = "send">发送事件</button>
    </div>
</template>

<script>
import emitter from './utils/eventbus.js'

export default {
    methods: {
        send() {
        	emmiter.emit('sendData', {name: "MXR", age: 22})
        }
    }
}
</script>


// 其他组件(兄弟/子孙)
<template>
	<div>
        <button @click = "send">发送事件</button>
    </div>
</template>

<script>
import emmiter from './utils/eventbus.js'

export default {
    methods: {
        created() {
            emitter.on("send", (info) => {		// 可通过emitter.on()监听多个事件
                console.log(info);				// 输出的info为事件发送时携带的值
            })
            
            emitter.on("*", (type, info) => {
                console.log(type, info)			// 通过*可以监听所有事件，type为事件的类型，info为携带的值
            })
        }
    }
}
</script>

// =================================================

<script>
    emitter.all.clear()			// 可通过emitter.all.clear()取消所有之前注册函数的监听
    
	function onFoo() {
		emitter.on('foo', onFoo)
        emitter.off('foo', onFoo)
	}
</script>

```

 
