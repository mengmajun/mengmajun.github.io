+++ 
date = '2025-07-01' 
draft = false 
title = '代码阅读：pure-vue-admin' 
categories = ['前端'] 
tags = ['pure-vue-admin'] 
+++

[github：vue-pure-admin](https://github.com/pure-admin/vue-pure-admin)
[文档地址](https://pure-admin.cn/pages/introduction/#%E5%9C%A8%E7%BA%BF%E9%A2%84%E8%A7%88)

## 修改主体内容区域背景色

src\style\sidebar.scss

```css
  .main-container {
    position: relative;
    height: 100vh;
    min-height: 100%;
    margin-left: $sideBarWidth;
    // 主体内容区域的背景颜色
    background: #ffffff;

    /* main-content 属性动画 */
    transition: margin-left var(--pure-transition-duration);

    .el-scrollbar__wrap {
      height: 100%;
      overflow: auto;
    }
  }
```

## 禁用导航栏面包屑导航、禁用阴影、系统配置、头像

src\layout\components\lay-navbar\index.vue

```html

// 获取用户名的第一个字符，如果没有用户名则显示"登"
const getFirstChar = computed(() => {
  return username || "登录";
});

<template>
  <div class="navbar bg-[#fff]">
    <LaySidebarTopCollapse
      v-if="device === 'mobile'"
      class="hamburger-container"
      :is-active="pureApp.sidebar.opened"
      @toggleClick="toggleSideBar"
    />

    <!-- <LaySidebarBreadCrumb
      v-if="layout !== 'mix' && device !== 'mobile'"
      class="breadcrumb-container"
    /> -->

    <LayNavMix v-if="layout === 'mix'" />

    <div v-if="layout === 'vertical'" class="vertical-header-right">
      <!-- 菜单搜索 -->
      <!-- <LaySearch id="header-search" /> -->
      <!-- 全屏 -->
      <!-- <LaySidebarFullScreen id="full-screen" /> -->
      <!-- 消息通知 -->
      <LayNotice id="header-notice" />
      <!-- 退出登录 -->
      <el-dropdown trigger="click">
        <!-- <span class="el-dropdown-link navbar-bg-hover select-none">
          <img :src="userAvatar" :style="avatarsStyle" />
          <p v-if="username" class="dark:text-white">{{ username }}</p>
        </span> -->
        <div
          class="grid place-items-center w-15 h-8 rounded-md bg-black text-white text-s font-bold"
        >
          {{ getFirstChar }}
        </div>
        <template #dropdown>
          <el-dropdown-menu class="logout">
            <el-dropdown-item @click="logout">
              <IconifyIconOffline
                :icon="LogoutCircleRLine"
                style="margin: 5px"
              />
              注销
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
      <!-- <span
        class="set-icon navbar-bg-hover"
        title="打开系统配置"
        @click="onPanel"
      >
        <IconifyIconOffline :icon="Setting" />
      </span> -->
    </div>
  </div>
</template>
```


## 调整侧边栏

- 去掉折叠箭头
- 增加升级套餐显示

```html
<template>
  <div
    v-loading="loading"
    :class="[
      'sidebar-container',
      showLogo ? 'has-logo' : 'no-logo',
      device === 'mobile' ? 'mobile-layout' : ''
    ]"
    @mouseenter.prevent="isShow = true"
    @mouseleave.prevent="isShow = false"
  >
    <LaySidebarLogo v-if="showLogo" :collapse="isCollapse" />

    <div class="sidebar-content">
      <el-scrollbar
        wrap-class="scrollbar-wrapper"
        :class="[device === 'mobile' ? 'mobile' : 'pc']"
      >
        <el-menu
          unique-opened
          mode="vertical"
          popper-class="pure-scrollbar"
          class="outer-most select-none"
          :collapse="isCollapse"
          :collapse-transition="false"
          :popper-effect="tooltipEffect"
          :default-active="defaultActive"
        >
          <LaySidebarItem
            v-for="routes in menuData"
            :key="routes.path"
            :item="routes"
            :base-path="routes.path"
            class="outer-most select-none"
          />
        </el-menu>
      </el-scrollbar>

      <!-- 固定在底部的升级套餐区域 -->
      <div
        class="absolute bottom-10 left-0 right-0 p-3 bg-gradient-to-r from-indigo-50 to-white border-t border-gray-200 transition-all duration-300 z-10"
      >
        <div class="flex items-center gap-2 w-full">
          <div
            class="flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-md bg-indigo-100 text-indigo-600"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="w-5 h-5"
              viewBox="0 0 24 24"
            >
              <path
                fill="currentColor"
                d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"
              ></path>
              <circle cx="11.5" cy="7.5" r="2.5" fill="currentColor"></circle>
            </svg>
          </div>
          <div class="flex-grow min-w-0">
            <div class="text-sm font-semibold text-gray-800 truncate">
              升级套餐
            </div>
            <div class="text-xs text-gray-500 truncate">解锁更多功能</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 折叠按钮现在位于升级区域下方 -->
    <LaySidebarLeftCollapse
      v-if="device !== 'mobile'"
      :is-active="pureApp.sidebar.opened"
      @toggleClick="toggleSideBar"
    />

    <!-- <LaySidebarCenterCollapse
      v-if="device !== 'mobile' && (isShow || isCollapse)"
      :is-active="pureApp.sidebar.opened"
      @toggleClick="toggleSideBar"
    /> -->
  </div>
</template>
```

## 修改侧边栏背景色

src\style\index.scss

```css
/* 自定义全局 CssVar */
:root {
  /* 左侧菜单展开、收起动画时长 */
  --pure-transition-duration: 0.3s;

  /* 常用border-color 需要时可取用 */
  --pure-border-color: rgb(5 5 5 / 6%);

  /* switch关闭状态下的color 需要时可取用 */
  --pure-switch-off-color: #a6a6a6;

  /** 主题色 */
  --pure-theme-sub-menu-active-text: initial;
  
  // 自定义菜单按钮背景颜色
  --pure-theme-menu-bg: #f9f9f9 !important;
  --pure-theme-menu-hover: none;
  --pure-theme-sub-menu-bg: transparent;
  --pure-theme-menu-text: initial;
  --pure-theme-sidebar-logo: none;
  --pure-theme-menu-title-hover: initial;
  --pure-theme-menu-active-before: transparent;

  // 修改全局的elementplus的颜色
  // --el-color-primary: #eae8e8 !important;
}

```


src\style\sidebar.scss

```css

 .sidebar-container {
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    z-index: 1001;
    width: $sideBarWidth !important;
    height: 100%;
    overflow: visible;
    font-size: 0;
    //自定义
    background: var(--pure-theme-menu-bg) !important;
    border-right: 1px solid var(--pure-border-color);


.el-menu .el-menu--inline .el-sub-menu__title,
    & .el-sub-menu .el-menu-item {
      min-width: $sideBarWidth !important;
      font-size: 14px;
      background-color: var(--pure-theme-menu-bg) !important;
    }
```

src\layout\components\lay-sidebar\components\SidebarLogo.vue

```css

<style lang="scss" scoped>
.sidebar-logo-container {
  position: relative;
  width: 100%;
  height: 48px;
  overflow: hidden;
  //自定义
  background: var(--pure-theme-menu-bg) !important;

```

## 侧边栏折叠Icon靠右

src\layout\components\lay-sidebar\components\SidebarLeftCollapse.vue

```css
<style lang="scss" scoped>
.left-collapse {
  position: absolute;
  bottom: 0;
  width: 100%;
  height: 40px;
  line-height: 40px;
  box-shadow: 0 0 6px -3px var(--el-color-primary);
  display: flex;
  justify-content: flex-end;
  padding-top: 12px;
  padding-right: 16px; // 添加右边距
}
</style>

```

## 主页面增加暗黑模式按钮



## 修改Button选中样式为黑色




## 修改为静态路由


## 修改进入主页面无需登录


## 实现登录页面

## 实现注册页面
