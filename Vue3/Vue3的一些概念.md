#### Vue.set()

可以向响应式对象内添加一个property，并确保该property同样是响应式的，并且触发视图更新。

注意，添加的对象不能为Vue实例，也不可是Vue实例的根数据（data）对象。



#### $attrs

​	使用$attrs可以指定父组件传入值绑定到具体哪一个根上(一般用于访问非props属性)。

```vue
<template>
	<test id = "test"></test>
</template>

<template>
	<h2>测试</h2>
    <h2 :id = "$attrs.id">测试</h2>	<!-- 此时控制台可以看到第二个h2的id为test -->
    <h2>测试</h2>
</template>
```





#### 子组件传值

​	在Vue3中子组件传值方式需要进行注册后才能使用。

```vue
<script>
	export default {
        data() {
            return {
                num: 0
            }
        }
        
        emit: ["add", "sub"]	<!-- 此处进行注册 -->
        
        emit: {					<!-- 这种对象写法可用于参数验证 -->
        	add: null,
        	sub: null,
        	addNumber: (num, name) => {
                console.log(num, name);
                if(num > 10) {
                    return true
                }
                return false	<!-- 返回false时会报警告，但仍然能传值 -->
            }
    	}
        
        methods: {
        	increment() {
                this.$emit("add")
            },
            decrement() {
                this.$emit("sub")
            },
            addNumber() {
                this.$emit("addSelf", this.num)
            }
    	}
    }
</script>
```



