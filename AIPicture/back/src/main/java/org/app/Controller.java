package org.app;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.google.gson.Gson;
import org.apache.tomcat.util.codec.binary.Base64;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import java.io.*;


@CrossOrigin(origins = {"*", "null"})
@RestController
@RequestMapping("/system/picture")
public class Controller {
    String path = "D:/AIPicture/OriginalPicture/OriginalPicture";
    @GetMapping("/downLoadFile")
    public void downLoadFile(@RequestParam("path") String path,@RequestParam("name") String name,HttpServletResponse response) throws Exception {
        new DownFileService().downLoadFile(path,name,response);
    }

    @PostMapping("/upload")
    public String upload(@RequestParam MultipartFile file){
        if (file.isEmpty()){
            return "upload fail";
        }
        String fileName = file.getOriginalFilename();
        String fileFormat = fileName.substring(fileName.lastIndexOf("."));
        try {
            file.transferTo(new File(path + fileFormat));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @GetMapping("/start")
    public boolean start(){
        new PythonAgent().startPython();
        return false;
    }
    @GetMapping("/resize")
    public boolean resize(){
        new PythonSRCNN().startPython();
        return false;
    }

}
