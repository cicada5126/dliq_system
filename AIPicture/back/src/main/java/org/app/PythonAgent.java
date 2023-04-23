package org.app;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PythonAgent {
    PythonAgent(){}

    public boolean startPython(){
        try {
            Process proc = Runtime.getRuntime().exec("C:\\Users\\86136\\anaconda3\\python.exe D:\\AIPicture\\back\\src\\main\\java\\org\\app\\Python\\pridnet\\testprid.py");// 执行py文件
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }


            in.close();
            proc.waitFor();
            return true;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        return false;
    }
}
