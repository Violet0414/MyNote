**Better-Scroll的使用：**

结构：

```html
 <div class="wrapper">
    <ul class="content">
      <li>...</li>
      <li>...</li>
      ...
    </ul>
  </div>
```

加载和实例化以及相关的方法：

```vue
<template>
  <div ref="wrapper">
    <slot></slot>
  </div>
</template>

<script>
  import BScroll from 'better-scroll'

	export default {
		name: "Scroll",
    props: {
		  probeType: {
		    type: Number,
        default: 1
      },
      data: {
		    type: Array,
        default: () => {
          return []
        }
      },
      pullUpLoad: {
		    type: Boolean,
        default: false
      }
    },
    data() {
		  return {
		    scroll: {}
      }
    },
    mounted() {
		  setTimeout(this.__initScroll, 20)
    },
    methods: {
		  __initScroll() {
		    // 1.初始化BScroll对象
		    if (!this.$refs.wrapper) return
        this.scroll = new BScroll(this.$refs.wrapper, {
          probeType: this.probeType,
          click: true,
          pullUpLoad: this.pullUpLoad
        })

        // 2.将监听事件回调
        this.scroll.on('scroll', pos => {
          this.$emit('scroll', pos)
        })

        // 3.监听上拉到底部
        this.scroll.on('pullingUp', () => {
          console.log('上拉加载');
          this.$emit('pullingUp')
        })
      },
      refresh() {
        // 重新计算scroll,防止当前高度发生变化后scroll还未记录
        this.scroll && this.scroll.refresh && this.scroll.refresh() 
      },
      finishPullUp() {
		this.scroll && this.scroll.finishPullUp && this.scroll.finishPullUp()
      },
      scrollTo(x, y, time) {
		this.scroll && this.scroll.scrollTo && this.scroll.scrollTo(x, y, time)
      }
    },
    watch: {
		data() {
        	setTimeout(this.refresh, 20)
      		}
    	}
	}
</script>

<style scoped>

</style>

```



**Better-Scroll原理**

如下图所示，wrapper为可滚动区域，content为当前页面内容。

  **注意，better-scroll初次进入页面后就会计算wapper区域和content区域的大小，若不使用refresh()进行刷新很有可能会使得滚动显示效果出问题。**



![](E:\笔记\笔记图片\Better-Scroll原理.png)





**监听滚动：**

const bscroll = new BScroll(元素, {此处传入属性})

bscroll.on('scroll', (position) =>{})     该方法返回的position就是当前页面滚动到的位置（传回坐标x和y的值）

probeType: 0/1/2(手指滚动)/3(只要是滚动都监听)





**refresh()方法：**

重新计算Better-Scroll，当DOM结构发生变化时务必要调用该方法以确保滚动效果的正常。

应用场景：需重新计算当前页面高度时。（类似于下拉加载更多数据）



**pullingUp 和pullingDown事件**：

触发时机：当距离滚动到底部小于 threshold 值时（上拉到底），触发一次 pullingUp 事件。

触发时机：当距离滚动到底部大于 threshold 值时（下拉到顶），触发一次 pullingDown事件。



**finishPullUp()方法和finishPullDown()方法：**

标识一次拉动动作的结束。在这两种方法被调用前，其对应的拉动事件只会触发一次。



**scrollTo(x, y, time)方法：**

x: 界面横坐标。

y: 界面纵坐标。

time: 完成该函数花费的时间。

应用场景： 需跳转到当前界面某一位置。（类似回到顶部的需求）



**钩子**
  BetterScroll 除了提供了丰富的 API 调用，还提供了一些事件，方便和外部做交互。你可以利用它们实现一些更高级的 feature。
  查看文档：API 钩子
  刚才上面finishPullDown()的例子中就用到了"pullingDown"事件，现在想在滚动过程中显示一行 “下拉刷新” 的文字，所以需要用到"scroll"这个事件：

**scroll**
**参数：{Object} {x, y} 滚动的实时坐标**
**触发时机：滚动过程中。**

```js
// 使用钩子scroll事件在下拉过程中显示一些信息，此处我想要显示“下拉刷新”
this.scroll.on("scroll",function(){
    // 不能用箭头函数，因为我们要用触发当前事件的对象this进行操作，所以不能改变this指向
    console.log(this.y);
    if(this.y>10){
        $(".posi_ref").show();
    }else{
        $(".posi_ref").hide();
    }
})
```





