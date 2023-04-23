<template>

  <div id="building" class="bg-white : text-center ">

      <strong>清图--照片修复系统</strong>
    <h1>

      <router-link to="/home"><el-button type="primary" plain>首页</el-button></router-link>
      <router-link to="/uploads"><el-button type="primary" plain>照片修复</el-button></router-link>
      <router-link to="/CompareImage"><el-button type="primary" plain>前后对比</el-button></router-link>


    </h1>

    <router-view></router-view>


  </div>




</template>

<script>


import Uploads from "@/components/Uploads";
import Home from "@/components/Home";
import CompareImage from "@/components/CompareImage";
export default {
  name: 'App',

  components:[
      Uploads,
      Home,
    CompareImage,
  ]
}
</script>

<style>
#building{
  background-image:url("http://localhost:8080/picture/R-C.jpg");
width:100%;
height:100%;
position:fixed;
background-size:100% 100%;
}
</style>

