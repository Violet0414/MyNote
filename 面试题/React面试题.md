### React18有哪些更新

1. 并发模式

2. 更新render API

3. 自动批处理

4. startTransition
5. useTransition
6. useDeferredValue
7. useId
8. 提供给第三方库的 Hook
9. Suspense 支持 SSR

--------------

### 简述React的生命周期

#### constructor():

​	在React组件挂载之前被调用，在为React.Component子类实现勾走函数时，应在其他语句之前调用super();

​	通常该函数仅用于以下两种情况：

1. 初始化函数内部的state
2. 为事件处理函数绑定实例

如果不初始化 `state` 或不进行方法绑定，则不需要写 `constructor()` , 只需要设置 `this.state` 即可。

不能在 `constructor()`构造函数内部调用 `this.setState()`, 因为此时第一次 `render()`还未执行，也就意味DOM节点还未挂载。



#### static getDerivedStateFromProps(nextProps, state)：

​	**该方法在调用render方法之前调用，在初始化和后续更新都会被调用。**

返回值：然会一个对象来更新state，如果返回null则不更新任何内容。

参数：第一个参数为即将更新的props，第二个参数为上一个状态的state，可以比较props和state来加一些限制条件，避免state无用的更新。



#### render():

​	该方法是类组件中为一个一必须实现的方法，用于渲染dom节点，render()方法必须返回reactDOM。



#### componentDidMount():

​	该函数在组件挂在后立即调用，**该函数是发送网络请求、启用事件监听方法的好时机**，并且在此钩子函数中可以直接调用setState()方法。



#### shouldComponentUpdate(nextProps，nextState)：

​	该函数在组件更新之前调用，可以控制组件是否进行更新，当返回true时组件更新，返回false则不更新。

1. 包含两个参数，第一个是即将更新的 props 值，第二个是即将更新后的 state 值，可以根据更新前后的 props 或 state 来比较加一些限制条件，决定是否更新，进行性能优化。
2. 不建议在 `shouldComponentUpdate()` 中进行深层比较或使用 `JSON.stringify()`。这样非常影响效率，且会损害性能。
3. 不要 `shouldComponentUpdate` 中调用 setState()，否则会导致无限循环调用更新、渲染，直至浏览器内存崩溃。
4. 可以使用内置 **[`PureComponent`](https://link.juejin.cn/?target=https%3A%2F%2Fzh-hans.reactjs.org%2Fdocs%2Freact-api.html%23reactpurecomponent)** 组件替代。



#### getSnapshotBeforeUpdate(prevProps, prevState):

​	该函数在最近一次的渲染输出被提交之前调用。即render()之后，即将对组件精选挂载时调用。

​	它可以使组件在 DOM 真正更新之前捕获一些信息（例如滚动位置），此生命周期返回的任何值都会作为参数传递给 `componentDidUpdate()`。如不需要传递任何值，那么请返回 null。



#### componentDidUpdate(prevProps, prevState, snapshot)：

​	该函数会在更新后立即调用，首次渲染不会执行。

​	包含三个参数，第一个是上一次props值。 第二个是上一次state值。如果组件实现了 `getSnapshotBeforeUpdate()` 生命周期（不常用），第三个是“snapshot” 参数传递。



#### componentWillUnmount()：

​	该函数在组件即将被卸载或销毁时进行调用。

​	**此生命周期是取消网络请求、移除监听事件、清理DOM元素、清理定时器等操作的好时机。**

------------

### 生命周期执行顺序

#### 创建时：

1. constructor();

     		2. static getDerivedStateFromProps();
     		3. render();
     		4. componentDidMount();

#### 更新时：

1. static getDerivedStateFromProps();

 	2. shouldComponentUpdate();
 	3. render();
 	4. getSnapshotBeforeUpdate();
 	5. componentDidUpdate();

#### 卸载时：

1. componentWillUnmount();

-------------------

### 各类小知识点

#### setState(newState, returnFun)

​	newState: 可以是一个对象或一个函数，用于描述状态的变化。

​	returnFun: 是一个可选的回调函数，在状态更新完成后会被调用。

------------

#### React.createElement和React.cloneElement的区别

#### **1. 用途**

- **`React.createElement`**:

  - 用于创建新的 React 元素。
  - 它可以接收一个组件类型（例如字符串标签或组件类）和一个 props 对象，以及子元素，然后返回一个新的 React 元素。

  ```jsx
  javascriptCopy Codeconst element = React.createElement('div', { className: 'my-div' }, 'Hello World');
  ```

- **`React.cloneElement`**:

  - 用于克隆已有的 React 元素，并可以对其进行修改（例如添加或修改 props）。
  - 它接收一个元素和一个新的 props 对象，然后返回一个新的元素，新的元素会基于原有元素的 props 和子元素。

  ```jsx
  javascriptCopy Codeconst originalElement = <div className="my-div">Hello World</div>;
  const clonedElement = React.cloneElement(originalElement, { className: 'new-class' });
  ```



#### 2. **参数**

- **`React.createElement`**:

  - **第一个参数**：组件类型（字符串标签或 React 组件）。
  - **第二个参数**：props 对象（可选）。
  - **后续参数**：子元素（可选，可以是任意数量）。

- **`React.cloneElement`**:

  - **第一个参数**：要克隆的 React 元素。
  - **第二个参数**：新的 props 对象（可选）。
  - **后续参数**：可以用来替代或添加新的子元素。

  

#### 3. **返回值**

- **`React.createElement`**:

  - 返回一个新的 React 元素，代表传入的组件及其 props 和子元素。

- **`React.cloneElement`**:

  - 返回一个新的元素，该元素是基于原始元素的拷贝，并应用了传入的新的 props。

  

#### 4. **示例**

#### 使用 `React.createElement`

```jsx
javascriptCopy Codeconst MyComponent = () => {
  return React.createElement('h1', { className: 'header' }, 'Hello, World!');
};
```

#### 使用 `React.cloneElement`

```jsx
javascriptCopy Codeconst MyComponent = ({ child }) => {
  const clonedChild = React.cloneElement(child, { className: 'new-class' });
  return <div>{clonedChild}</div>;
};

// 使用示例
const App = () => {
  return (
    <MyComponent child={<div className="original-class">Hello, World!</div>} />
  );
};
```

#### 总结

- **`React.createElement`** 主要用于创建新的元素，而 **`React.cloneElement`** 主要用于克隆现有的元素并可以修改其 props。
- 这两个方法在 React 的元素处理和组件组合中都非常有用，可以根据需要选择使用

--------

#### useEffect和useLayoutEffect的区别

#### 1. **执行时机**

- **`useEffect`**：

  - 在组件渲染到屏幕后执行。具体来说，它在浏览器完成绘制后运行，这意味着用户会先看到页面的更新，然后再执行 `useEffect` 中的代码。这种行为有助于避免阻塞浏览器的绘制。

- **`useLayoutEffect`**：

  - 在浏览器绘制之前执行。在所有 DOM 变更之后、浏览器实际绘制之前，它会立即运行。这意味着，如果在 `useLayoutEffect` 中有 DOM 操作，这些操作会在用户看到变化之前完成，从而避免了闪烁或视觉上的不一致。

  

#### 2. **性能影响**

- **`useEffect`**：

  - 由于它在绘制之后执行，通常不会影响性能和用户体验。你可以安全地进行数据获取、事件监听等操作，而不担心会导致视觉延迟。

- **`useLayoutEffect`**：

  - 因为它会在浏览器绘制之前运行，如果它执行了耗时的操作，可能会导致性能下降和页面卡顿。因此，通常建议仅在需要同步更新 DOM 或读取布局信息时使用。

  

#### 3. **使用场景**

- **`useEffect`**：

  - 适用于大多数副作用场景，例如数据获取、订阅、设置事件监听器等。因为它不会阻塞浏览器的绘制，所以一般推荐使用 `useEffect`。

- **`useLayoutEffect`**：

  - 适用于需要在浏览器绘制之前同步执行副作用的场景，比如读取 DOM 元素的大小、滚动位置等，或者直接在 DOM 上进行修改以避免用户看到不一致的状态。

  

#### 4. **执行顺序**

- 当组件更新时，`useLayoutEffect` 会在 `useEffect` 之前执行。这意味着在 `useLayoutEffect` 完成之前，任何新的 DOM 变更都不会被绘制到屏幕上。

### 代码示例

```jsx
import React, { useEffect, useLayoutEffect } from 'react';

function MyComponent() {
  useEffect(() => {
    console.log('useEffect: Component rendered');
    // 可以进行数据获取、订阅等操作
  }, []);

  useLayoutEffect(() => {
    console.log('useLayoutEffect: Component rendered');
    // 适合需要同步读取布局并且影响显示的操作
  }, []);

  return <div>My Component</div>;
}
```

----------------

### Redux状态管理库

#### 主要概念

1. **Store**:

   - Redux 使用一个单一的状态树，称为 **store**，来存储整个应用的状态。
   - 这个状态是只读的，唯一可以改变状态的方式是通过 **dispatch** 发送动作（action）。

2. **Action**:

   - **action** 是一个普通的 JavaScript 对象，用于描述状态的变化。每个 action 至少需要有一个 `type` 属性来标识其类型。

   - 示例：

     ```jsx
     const ADD_TODO = 'ADD_TODO';
     const addTodoAction = {
       type: ADD_TODO,
       payload: { text: 'Learn Redux' }
     };
     ```

3. **Reducer**:

   - **reducer** 是一个纯函数，用于处理 action 并返回新的状态。

   - Reducer 接受当前状态和 action 作为参数，并返回更新后的状态。

   - 示例：

     ```jsx
     const initialState = { todos: [] };
     
     const todoReducer = (state = initialState, action) => {
       switch (action.type) {
         case ADD_TODO:
           return { ...state, todos: [...state.todos, action.payload] };
         default:
           return state;
       }
     };
     ```

4. **Dispatch**:

   - **dispatch** 是一个方法，用于发送 action 到 store，触发 reducer 来更新状态。

   - 示例：

     ```jsx
     store.dispatch(addTodoAction);
     ```

5. **Selector**:

   - **selector** 是一个函数，用于从 store 中获取特定的状态部分，通常用于优化和封装 state 访问的逻辑。

   

#### Redux 的特点

1. **单一数据源**:

   - Redux 维护一个单一的 store，使得所有的状态集中管理，这对于调试和追踪状态变化非常有利。

2. **不可变状态**:

   - Redux 强调状态的不可变性，每次状态的变化都会返回一个新的状态对象，而不是直接修改原有状态。

3. **中间件**:

   - Redux 支持中间件，例如 redux-thunk 或 redux-saga，可以用于处理异步操作和 side effects。

4. **可预测性**:

   - 由于状态的变化只能通过 actions 和 reducers 来实现，Redux 的状态管理变得非常可预测。

5. **开发者工具**:

   - Redux 提供了强大的开发者工具，可以追踪状态变化、时间旅行调试等，极大地方便了开发和调试过程。

   

#### 使用示例

以下是一个简单的 Redux 使用示例：

```jsx
import { createStore } from 'redux';

// Action
const ADD_TODO = 'ADD_TODO';

const addTodo = (text) => ({
  type: ADD_TODO,
  payload: { text }
});

// Reducer
const initialState = { todos: [] };

const todoReducer = (state = initialState, action) => {
  switch (action.type) {
    case ADD_TODO:
      return { ...state, todos: [...state.todos, action.payload.text] };
    default:
      return state;
  }
};

// Create Store
const store = createStore(todoReducer);

// Dispatch Action
store.dispatch(addTodo('Learn Redux'));
console.log(store.getState()); // { todos: ['Learn Redux'] }
```

---------------



















