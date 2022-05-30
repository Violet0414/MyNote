#### axios封装

​	使用以下方式可以便于在各种环境下进行网络请求。

```typescript
// server/request/index.ts
import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig } from 'axios'

class MyRequest {
	instance: AxiosInstance;										// axios实例，调用该实例的request方法
  constructor(config: AxiosRequestConfig) {		// 构造器
    this.instance = axios.create(config);
  }

  request(config: AxiosRequestConfig) {
    // 此处相当于axios.request()
    this.instance.request(config).then((res) => {
      console.log(res.data.data);
    });
  }
}

export default MyRequest;

// server/index.ts 出口文件
import MyRequest from './request/index'

const myRequest = new MyRequest({
  baseURL: 'http://123.207.32.32:8000',
  timeout: 10000,
})

export default myRequest;


// 使用文件
import myRequest from "xxxxxx/server/index"

myRequest.request({
  url: "/home/multidata",
  method: "GET",
})


```





#### axios封装思想进阶

```typescript
// type.ts  该文件为ts类型的声明抽取
import { AxiosRequestConfig, AxiosResponse } from "axios";

export interface MyRequestInterceptors {
  requestInterceptors?: (config: AxiosRequestConfig) => AxiosRequestConfig;
  requestInterceptorsCatch?: (error: any) => any;
  responseInterceptors?: (res: AxiosResponse) => AxiosResponse;
  responseInterceptorsCatch?: (error: any) => any;
}

export interface MyRequestConfig extends AxiosRequestConfig {
  interceptors?: MyRequestInterceptors;
}

// index.ts		此文件为instance实例以及构造器的声明
import axios from "axios";
import type { AxiosInstance, AxiosRequestConfig } from "axios";
import type { MyRequestInterceptors, MyRequestConfig } from "./type";

class MyRequest {
  instance: AxiosInstance;
  interceptors?: MyRequestInterceptors;

  constructor(config: MyRequestConfig) {
    this.instance = axios.create(config);

    this.interceptors = config.interceptors;
    this.instance.interceptors.request.use(
      this.interceptors?.requestInterceptors,
      this.interceptors?.requestInterceptorsCatch
    );
    this.instance.interceptors.response.use(
      this.interceptors?.responseInterceptors,
      this.interceptors?.responseInterceptorsCatch
    );

    // 为所有实例添加一个共有拦截器，共有的拦截器
    this.instance.interceptors.request.use(
      (config) => {
        console.log("共有请求拦截成功");
        return config;
      },
      (err) => {
        return err;
      }
    );

    this.instance.interceptors.response.use(
      (res) => {
        console.log("共有响应拦截成功");
        return res;
      },
      (err) => {
        return err;
      }
    );
  }

  request(config: AxiosRequestConfig) {
    this.instance.request(config).then((res) => {
      console.log(res.data.data);
    });
  }
}

export default MyRequest;

// 最外层的暴露文件
import MyRequest from "./request/index";

const myRequest = new MyRequest({
  baseURL: "http://123.207.32.32:8000",
  timeout: 10000,
  interceptors: {
    requestInterceptors: (config) => {
      console.log("请求成功了");
      return config;
    },
    requestInterceptorsCatch: (err) => {
      console.log("请求失败了");
      return err;
    },
    responseInterceptors: (res) => {
      console.log("响应成功了");
      return res;
    },
    responseInterceptorsCatch: (err) => {
      console.log("响应失败了");
      return err;
    },
  },
});

export default myRequest;


// main.ts
import "./server/axios";
import myRequesy from "./server/index";

myRequesy.request({
  url: "/home/multidata",
  method: "GET",
});
```

