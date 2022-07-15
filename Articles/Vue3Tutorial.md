---
title: Vue3 001
tags:["tutorial"]
excerpt:Vue3 001
date: 2022-07-15
---

```javascript
// 所有使用的钩子都要引入
import { 
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted,
  onRenderTracked,
  onRenderTriggered,
  watch,
  computed
} from 'vue'
export default {
  name: 'App',
  components: {},
  props: {},
// 这里的props不能使用解构，否则会不响应，如果用解构可以用toRef使之响应
  setup(props, context) {
    // 所有的钩子函数都写进这个里面，在外面写也会执行，是向下兼容的，但是不建议两个都写
    console.log('这里的setup函数执行就相当于vue2的钩子beforeCreate和created')
    // 所以在这里this是不能用的，因为setup在一开始就运行了
    // 其他钩子XXX的执行需要在这用on前缀写上:onXXX(()=>{}),相当于执行的是一个方法，传入的参数是个回调函数
    onBeforeMount(()=>{
      console.log('setup的钩子和vue2.0的钩子同时存在的时候，setup里的钩子会先执行')
    }) // 这里不要习惯性的写逗号，这里是调用钩子，而不是options
    onMounted(()=>{
      console.log('渲染完了')
    })
    onBeforeUpdate(()=>{
      console.log('更新前')
    })
    onUpdated(()=>{
      console.log('更新后')
    })
// 大神说unmount（卸载）要比destroy（销毁）要更形象些，所以这个版本换个名字
    onBeforeUnmount(()=>{
      console.log('卸载前：vue3.0的这个生命周期命名要区别于vue2.0的beforeDestroy')
    })
    onUnmounted(()=>{
      console.log('卸载后：vue3.0的这个生命周期命名要区别于vue2.0的destroyed')
    })
    onRenderTracked((event) => {
      console.log('新增的钩子状态跟踪函数（跟踪全部的，需要自己去查看哪个变化了）：其实就是跟踪return 里数据的变化')
    })
    onRenderTriggered(event => {
      console.log('新增的钩子状态跟踪函数(区别在与精确跟踪变化的那个数据)')
    })
    // 监听一个值，如果要监听多个值，第一个参数就是个数组，元素是要监听的多个值
    watch('要监听的值', (val, oldVal) => {})
    // 计算属性：创建一个响应式的str计算属性，根据依赖的ref自动计算并返回一个新的ref
    const i : number = ref(0)
    const  str = computed(() => i.value++;
    return {
      i，
      str 
    }
}



import { ref, toRefs, reactive }  from 'vue'
export default {
  name: 'Test',
  setup(){
    // 在setup中定义template模板使用的响应式变量，你得用ref或者reactive来定义
    // 这里的ref你可以理解成一个工厂类，传入的参数就是初始化的变量的值，跟组件的ref概念不能混淆
    // 定义单个变量，为了让大家明白意义，我跟vue2.0都进行下对比 
    // vue2.0,不管定义单个或者多个我们都是定义 在data里，如
    /*
       data(){
          return {
            name: '小哥哥'
          }
       }
    */
    // 在vue3.0中，如果你只需要定义一个响应式变量，那么你可以用以下ref
    // 可能你会疑惑既然是定义变量为什么不用let，var，而用const定义常量的，这里是因为你定义的是一个引用，name指向的永远是一个固定不变的指针地址
    const name = ref('小哥哥')
    // 注意点，这里要获取name的值怎么获取,通过定义的变量的。value
    console.log('拿到name的值：', name.value)
    // 检测某个值是不是响应式的可以用isRef
    
    // 每次都用.value去拿值的写法，是不是有点不适应，而且定义的变量多了我们也不可能定义一堆ref，看起来都丑
   // reactive 可以创建一个响应式数据，参数是一个对象
    const data = reactive({
       name: '帅的没人要'，// 这样创建的响应式数据就不用ref了，写起来是不是要方便点
        sex: '性别分不出'，
        arr: []
    })
    // 但是以上还是有个缺点，如果你在return里直接放上data,那你使用的时候每次都要data.name，data.sex也麻烦，<div>{{data.sex}}</div>
   // 你说你可以解构在return {...data} 如果你这样的，里面的数据都会变成非响应式的，vue3.0提供了一个满足你幻想的方法toRefs,使用了这个包装下，你就可以在return里使用解构了
 // 包装上面的data
  const  refData = toRefs(data)
    // 在data中都有个return ，这里当然也必须要有的，但是这里是所有方法计算属性都要通过这个返回
    // 有人疑惑，那我可以直接在这个return上定义变量吗，答案是可以，但是定义的变量不是响应式的，不可变化
    return {
      ...refData, // 你也可以直接在这里用...toRefs(data) 这样会简单点
      name,
      rules: [] //如果你有什么不会变化的数据，如规则啊，配置啊，可以直接在这定义而不需要通过ref和reactive
    }
    
  }
}

import { ref } from 'vue'
export default {
  name: 'Test',
  setup(){
    // 定义一个响应式数据
    const baby = ref('1岁bb')
   // 定义method,把方法名字在return里返回
   // 注意：这里用调用响应式的数据也就是您定义的vue2.0的data,不可以用this,这个setup函数在01里已经说明过了，这个时候相当于vue2.0的beforeCreate和created，这里一开始就会运行，还没有this的指向，是个undefined，访问所有你定义的响应式的变量都要.value去访问
    const wantToKnowBabysName = () => {
      console.log("oh,he is a " + baby.value)
    }
    const getData = () => {}
   // 对比vue2.0
   /*
   method: {
      wantToKnowBabysName(){
        console.log("oh,he is a " + baby.value)
      },
      getData() {
      }
    }
  */
   
    return {
      baby,
      wantToKnowBabysName,
      getData
    }
    
  }
}

// 注意使用的时候引入computed
import { ref, computed} from 'vue'
export default {
  name: 'Test',
  setup(){
    // 定义一个响应式数据
    const baby = ref('嘎嘎嘎')
    // 定义翠花年龄
    const age = ref(28)
    // computed计算属性传入一个回调函数
    const areYouSureYouAreABaby = computed(() => {
      return `I'm sure,I'm a 300 month baby, ${baby.value}`
    })
    // set和get方式
    const setAge= computed({
      get() {
        return age.value + 10
      },
      set(v) {
        age.value = v - 10
      }
    })
   // 对比vue2.0
   /*
   computed: {
      areYouSureYouAreABaby (){
        return `I'm sure,I'm a 300 month baby, ${baby.value}`
      },
      setAge:{
        get(){
          return age + 10
        },
        set(v) {
          age = v - 10
        }
      }
    }
  */
   
    return {
      baby,
      age,
      areYouSureYouAreABaby 
    }
    
  }
}

// 注意使用的时候引入watch
import { ref, watch, watchEffect } from 'vue'
export default {
  name: 'Test',
  setup(){
    // 定义一个响应式数据
    const baby = ref('嘎嘎嘎')
    const arr = ref(['翠花', '小红'])
    // 监听一个值的情况，有两种方式
    // watch 有三个参数：第一个是个getter（所谓getter写法就是你要写个getter函数）,第二个是个回调函数，第三个是个options(这个参数是放vue2.0的deep或者immediate等可选项)
    // 第一种：直接放ref
    watch(baby, () => {
      return `I'm sure,I'm a 300 month baby, ${baby.value}`
    })
   // 第二种：放ref的value值
   watch(() => baby.value, () => {
      return `I'm sure,I'm a 300 month baby, ${baby.value}`
    })
  
   // 监听多个值的时候 ,第一个参数是个数组，里面放监听的元素
   watch([baby,arr], (v, o) => { 
     // 这里的v,o也是数组，所以你取值的时候v[0],v[1]拿到第几个元素的变化
     ...
   }, {
    deep: true,
    immediate: true
   })
 // 或者写成
 watch([baby,arr], ([baby, arr], [prebaby,prearr]) => {
    ...
  })
   // 对比vue2.0
   /*
   watch: {
      baby(v, o) {
        
      },
      arr: {
        handle(v,o) {},
        deep: true,
        immediate: true,
        flush: 'pre' // 这个默认有三个参数，'pre'| 'post' | 'sync'，默认‘pre’组件更新前运行,'post'组件渲染完毕后执行，一般用于你需要去访问$ref的时候可以用这个，'sync'是一旦你的值改变你需要同步执行回调的时候用这个
      }
    }
  */
   // watch的写法跟vue2的$watch写法一样，可以参考
  // vue3.0 watchEffect 用法
  //  watchEffect 有两个参数，一个是副作用函数(就是外部的数据对这个函数产生影响的，通俗点说就是在这个函数内部使用了外面的变量等)，一个是options（）
//  在vue2.0中，我们一般在created里添加一些监听事件，比如你的$bus的一些事件监听，在setup中就可以在这个里面写
watchEffect((onInvalidate) => {
   // 这里的变化就相当于依赖了age.value，如果age变化了就会触发这个监听
   // 刚刚创建组件的时候会立即执行这个 
   const _age= `her age is ${age.value}`
   console.log(_age)
   //有时候你需要在这里挂载一些监听事件
   const handerClick = ()=>{}
   document.addEventlistener('click', handerClick)
   // 在vue2.0我们需要在destroy的时候remove它，这里提供了一个方法onInvalidate回调解决remove的问题
   onInvalidate(()=>{
       /*
        执行时机:  在副作用即将重新执行时，就是在每次执行这个watchEffect回调的时候会先执行这个,如果在setup()或生命周期钩子函数中使用watchEffect, 则在卸载组件时执行此函数。
       */
       document.removeEventListener('click',handerClick )
    })  
})
// 这个也是支持async,await的
const data = ref(null)
watchEffect(async onInvalidate => {
 // 假设个接口获取数据的
  data.value = await fetchData()
  onInvalidate(() => {...})
})
// 再来理解options：这里有三个参数flush,onTrigger,onTrack
watchEffect(onInvalidate => {
  onInvalidate(() => {...})
}, {
  flush: 'pre' // 跟watch一样，默认pre，组件更新前去调用
  onTrigger(e) {
    // 依赖项变化时候触发这个即依赖项的set触发的时候
  },
  onTrack(e) {
    // 依赖项被调用的时候触发这个即依赖项的get触发的时候
  }
})
    return {
      baby,
      areYouSureYouAreABaby,
      data 
    }
    
  }
}

export default {
  name: 'Test',
  props: ['name', 'age'],
  // setup(props, context) { // 有的时候会这样写，你可能只用得到emit
  setup(props,{attrs, slots, emit}) // 如果你都用得到你可以这样解构的写出来，这个不是响应式的，所以可解构
   // 错误写法 const {name} = props 这里我理解你肯定想直接就使用name，age等
   // 这个props是一个响应式的Proxy对象，不可以解构，解构后悔失去响应，如果要用解构的方式，要用toRefs
    let { name, age } = toRefs(props)
    // 现在是不是感觉可以直接就用操作name和age了，天真，转成ref了，所有的访问值都要xx.value
   console.log(name.value,age.value)
   // 所以倒回去，是不是觉得还不如直接用props.name直接访问代理对象的值要好点
   console.log(props.name, props.age)

   // context 
   // 看到这个context的参数你应该知其意了撒
   //  attrs: 相当于vue2.0的$attrs,意思就是传进来的属性值除了props接受的那部分
   // slots: 就是插槽，你要获取插槽的什么值的话可以用这个
   // 插槽这里解释下，可能有部分人不知道咋拿值，顺便说下，这里vue3.0的所有响应式数据都是Proxy对象，所以你取值的时候都是proxy.xxx获取
   slots.default(args => {
    console.log('这里就是你在vue2.0里看到的所有slot的数组，就可以取你要哪个插槽的值了', args)
   })
   // emit: 这个是vue2.0 的$emit
   emit('方法名', '参数') // vue2.0 this.$emit('方法名', '参数')
    ...
  }
}

// 老父亲组件
import { provide, ref, reactive } from 'vue'
export default {
  name: 'Test',
  setup() {
    // 用法: provide(key, value) 用下面的ref和reactive是为了让数据变成响应式的，父组件变化，子组件数据跟着变
    const name = ref('小哥哥')
    const obj = reactive({
      name: '土狗',
      age: '3岁'
    })
    provide('name', name)
    provide('animal', obj)
  }
}
// 乖儿孙组件
import { inject } from 'vue'
export default {
  name: 'Child',
  setup() {
    // 用法: inject(key) 
    const name = inject('name')
    const animal = inject('animal')
    return {
      name,
      animal
    }
  }
}
```