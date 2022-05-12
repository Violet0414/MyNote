### Vuex

```js
import { createStore } from 'vuex'

const store = createStore({
    state() {
        return {
            name: "MXR",
            age: 22,
            books: [
                {name: "测试书籍1", price: 120, count: 1}，
                {name: "测试书籍2", price: 80, count: 3}，
                {name: "测试书籍3", price: 150, count: 6}，
            ],
            discount: 0.8,
            number: 100,
        }
    },
    
    getters: {
        totalPrice(state, getters) {						// 传入参数为需要使用的Vuex属性模块名称
            let totalPrice = 0;
            for(const book of state.books) {
                totalPrice += book.count * book.price
            }
            return totalPrice * getters.getDiscount			// 传入后可调用其内部相应属性或方法
        },

		getDiscount(state) {
    		return state.discount * 0.9
		},
        
        getCount(state, getters) {
            return function(n) {
                let totalNum = 0;
                for(const book of state.books) {
					if(book.count > n){
                        totalNum += book.count
                    }
                }
                return totalNum
            }
        },
        
        nameInfo(state) {					// 可用于数据的处理，例如拼接字符串
            return `姓名：${state.name}`
        },
        ageInfo(staet) {
            return `年龄：${state.age}`
        }
    },
    
    mutations: {
        increment(state) {
            state.number ++;
        },
        decrement(state) {
    		state.number --;
		},
        
        // payload为传值时所携带的参数（载荷）
        changeNumber(state, payload) {
            state.number += n;
        },
        // 其中payload也可以传对象，通过以下方式获取对象的值
        upPayload(state, payload) {
            console.log(payload.name);
            console.log(payload.age);
        }
    },
    
    action: {
        incrementAction(context, payload) {
            console.log(payload)		// 输出结果为count: 100,其和mutations类似
            settimeout(() => {
            	context.commit('increment')  
            }, 1000);
        }
    }
})

export default store;
```





#### state取值

```vue
<template>
	<div>
        <h2>{{ $store.state.name }}</h2>
     	<h2>{{ fullName }}</h2>
        <h2>{{ name }}</h2>
        <h2>{{ age }}</h2>
        <h2>{{ sName }}</h2>
        <h2>{{ sAge }}</h2>
    </div>
</template>

<script>
	import {mapState} from 'vuex'		// 对state进行映射的一个辅助函数
    
    export default {
        computed: {
            fullName() {
                return this.$store.name
            },
            
            // 使用展开运算符展开mapState，将mapState这个对象本身映射的属性放入computed对象内
            ...mapState(["name", "age"])
            
            // 对象写法，可自定义名称
            ...mapState({
            	sName: state => state.name,
            	sAge: state => state.age,
        	})
        }
    }
</script>
```

​	

#### getters使用

```vue
<template>
	<div>
        <h2>{{ $store.getters.totalPrice }}</h2>
        // 获取书架内数量大于3的书的总本书，虽然功能很奇怪，但就是个例子，若括号内不加数字则返回的为一个函数
        <h2>{{ $store.getters.getCount(3) }}</h2>
        
        <h2>{{ nameInfo }}</h2>
        <h2>{{ ageInfo }}</h2>
        
        <h2>{{ sName }}</h2>
        <h2>{{ sAge }}</h2>
    </div>
</template>

<script>
import { mapGetters } from 'vuex'   

export default {
	computed: {
        ...mapGetters["nameInfo", "ageInfo",]		// 也是辅助函数
        
        // 对象写法，起别名
        ...mapGetters({
        	sName: "nameInfo",
           	sAge: "ageInfo",
       	})     
    }
}
</script>
```



#### mutations(更改store中状态的唯一方法)

```vue
<template>
	<div>
        <h2>当前计数：{{ $store.state.number }}</h2>
        <hr>
        	<button @click="$store.commit('increment')">+1</button>
        	<button @click="$store.commit('decrement')">-1</button>
        	<button @click="increment">+1</button>
        	<button @click="decrement)">-1</button>
        
        	<button @click="changeNum">changeNum</button>
        	<button @click="payload">payload</button>
        <hr>
    </div>
</template>

<script>
import { mapMutations } from 'vuex'   

export default {
	data() {
        return {
            num: 10,
        }
    },
    
    methods: {
        changeNum() {
            this.$store.commit('changeNumber', this.num)	// 改变值为num的值
        },
        
        payload() {
            this.$store.commit('upPayload', {name: 'FXJN', age: 22})
        }
        
        // 辅助函数写法
       	...mapMutations["increment", "decrement"]
    }
}
</script>
```



#### action的基本使用

​	action类似于mutation，action内部也需要通过commit改变状态，但其可以包含任意异步操作。

```vue
<template>
	<div>
        <h2>当前计数：{{ $store.state.number }}</h2>
        <hr>
        	<button @click="increment">+1</button>
        <hr>
    </div>
</template>

<script>
import { mapMutations } from 'vuex'   

export default {
	data() {
        return {
            num: 10,
        }
    },
    
    methods: {
		increment() {
            // 分发action内的函数，通过action内的函数执行调用mutation内的方法实现效果，一般用于异步操作的实现
            this.$store.dispatch("incrementAction", {count: 100})
        }
    }
}
</script>
```



#### modules模块管理

​	将数据分出去细化，便于管理每个模块的数据，可以使得模块拥有自己的state、getters、actions、mutations甚至是modules。

```js
// Vuex子模块
const adminMoudle = {
    state() {
    	return {
      		adminCounter: 100	      
        }
    },
    getters: {
    
	},
    mutations: {
        
    },
    actions: {
        
    },
}

export default adminMoudle;
```

```js
// Vuex
import {createStore} from 'vuex'
import adminModules from './modules/admin'
import userModules from './modules/user'

const store = createStore({
    state() {
        return {
            rootCounter: 0
        }
    },
    modules: {
        admin: adminModules,
        user: userModules,
    }
})

export default store;
```

```vue
<template>
	<div>
        <h2>当前根计数：{{ $store.state.rootCounter }}</h2>
        <h2>当前管理员计数：{{ $store.state.admin.adminCounter }}</h2>
        <h2>当前用户计数：{{ $store.state.user.userCounter }}</h2>
    </div>
</template>

<script> 

export default {

    
    methods: {

    }
}
</script>
```

