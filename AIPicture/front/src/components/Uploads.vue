<template>
  <div>
    <div><h5>修复前vs修复后</h5>
      <img :src="imgUrl" class="leftimg" />

    <img src='http://localhost:8080/picture/OriginalPicture_prid.jpg'  class="rightimg"></div>
    <div style="margin-top:20px;" >
    <el-button size="medium" type="warning " plain @click="dialogVisible = true">此处上传</el-button>
    <el-button size="medium" type="plain" plain @click="rerender">智能降噪</el-button>
    <el-button size="medium" type="plain " plain @click="resize">无损放大</el-button>
      <p></p>
      <p>
    <el-button size="medium" type="success" plain @click="downLoadFile">下载降噪结果</el-button>
      <el-button size="medium" type="success" plain @click="downLoadFile">下载放大结果</el-button>
      </p>
    </div>
    <el-dialog
        title="提示"
        :visible.sync="dialogVisible"
        width="30%"
        :before-close="handleClose">
      <el-upload v-show="IsShow"
                 class="upload-demo"
                 drag
                 action="http://localhost:8080/system/picture/upload"
                 limit="1"
                 :on-success = "handleSuccess"
                 accept="jpg : png">
        <i class="el-icon-upload"></i>
        <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
        <div class="el-upload__tip" slot="tip">只能上传jpg/png文件</div>
      </el-upload>
      <span slot="footer" class="dialog-footer">
    <el-button @click="dialogVisible = false">取 消</el-button>
    <el-button type="primary" @click="forceRerender">确 定</el-button>
  </span>
    </el-dialog>

  </div>
</template>

<script>

import axios from "axios";

export default {
  // eslint-disable-next-line vue/multi-word-component-names
  name: "Uploads",
  data() {
    return {
      componentKey: 0,
      IsShow: true,
      dialogVisible: false,
      imgUrl:'http://localhost:8080/picture/OriginalPicture.png',

      filePath: 'D:/AIPicture/OriginalPicture/OriginalPicture_prid_srcnn_x3.jpg', // 下载文件路径
      fileName: 'OriginalPicture_prid_srcnn_x3.jpg', // 文件名称
      // fileName: 'Picture.png', // 文件名称


    };
  },
  methods: {


      downLoadFile() {
        axios({
          url:"http://localhost:8080/system/picture/downLoadFile",
          method:"GET",
          params: {
            path: this.filePath,
            name: this.fileName
          },
          responseType: 'blob'
        }).then(res => {
          const blob = new Blob([res.data]);
          const fileName = res.headers["content-disposition"].split(";")[1].split("filename=")[1];
          if ('download' in document.createElement("a")) {
            const link = document.createElement("a");
            link.download = fileName;
            link.style.display = 'none';
            link.href = URL.createObjectURL(blob);
            document.body.appendChild(link);
            link.click();
            URL.revokeObjectURL(link.href);
            document.body.removeChild(link);
          } else {
            navigator.msSaveBlob(blob, fileName);
          }
        })
      },

      forceRerender() {
      this.dialogVisible = false;
      this.componentKey += 1; // 或者 this.componentKey = new Date();
    },

    rerender(){
      axios({
        url:"http://localhost:8080/system/picture/start",
        method:"GET",
      }).then(res=>{
        console.log(res);
      })
      window.location.reload();
    },

    resize(){
      axios({
        url:"http://localhost:8080/system/picture/resize",
        method:"GET",
      }).then(res=>{
        console.log(res);
      })
      window.location.reload();
    },

    handleSuccess(file) {
      this.imgUrl = file.url;
    },
    handleClose(done) {
      this.$confirm('确认关闭？')
          // eslint-disable-next-line no-unused-vars
          .then(_ => {
            done();
          })
          // eslint-disable-next-line no-unused-vars
          .catch(_ => {});
    }
  }
}
</script>

<style >
.leftimg {
  height: 400px;
  wight: 400px;
  max-width: 100%;
  max-height: 100%;

}

.rightimg {
  height: 400px;
  wight: 400px;
  max-width: 100%;
  max-height: 100%;

}

</style>