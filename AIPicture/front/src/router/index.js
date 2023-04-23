import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from "@/components/Home";
import Uploads from "@/components/Uploads";
import CompareImage from "@/components/CompareImage";
Vue.use(VueRouter)

const router = new VueRouter({
    routes:[
        {path: '/',redirect: 'home'},
        {path: '/home', component: Home },
        {path: '/uploads', component: Uploads },
        {path: '/CompareImage', component: CompareImage },
    ]
})

export default router